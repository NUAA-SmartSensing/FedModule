import threading
import ctypes
import inspect
import torch.cuda
from fedasync import AsyncClientManager
from fedasync import CheckInThread
from fedasync import SchedulerThread
from fedasync import UpdaterThread
from utils import ModuleFindTool, Queue, Time


# 强制关闭线程
def _async_raise(tid, exc_type):
    """raises the exception, performs cleanup if needed"""
    try:
        tid = ctypes.c_long(tid)
        if not inspect.isclass(exc_type):
            exc_type = type(exc_type)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exc_type))
        if res == 0:
            # pass
            raise ValueError("invalid thread id")
        elif res != 1:
            # """if it returns a number greater than one, you're in trouble,
            # and you should call it again with exc=NULL to revert the effect"""
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")
    except Exception as err:
        print(err)


class SyncServer:
    def __init__(self, global_config, server_config, client_config, manager_config):
        # 全局模型
        model_class = ModuleFindTool.find_class_by_string("model", server_config["model_file"], server_config["model_name"])
        self.server_network = model_class()
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.server_network = self.server_network.to(self.dev)

        # 数据集
        dataset_class = ModuleFindTool.find_class_by_string("dataset", global_config["data_file"], global_config["data_name"])
        self.dataset = dataset_class(global_config["client_num"], global_config["iid"])
        self.test_data = self.dataset.get_test_dataset()
        self.T = server_config["epochs"]

        # 运行时变量
        self.current_t = Time.Time(0)
        self.queue = Queue.Queue()
        self.accuracy_list = []
        self.stop_event = threading.Event()
        self.stop_event.clear()
        self.server_thread_lock = threading.Lock()

        init_weights = self.server_network.state_dict()
        datasets = self.dataset.get_train_dataset()

        self.async_client_manager = AsyncClientManager.AsyncClientManager(init_weights, global_config["client_num"],
                                                                          datasets, self.queue, self.current_t,
                                                                          self.stop_event, client_config, manager_config)
        self.scheduler_thread = SchedulerThread.SchedulerThread(self.server_thread_lock, self.async_client_manager,
                                                                self.queue, self.current_t, server_config["scheduler"],
                                                                server_config["checkin"], self.server_network, self.T)
        self.updater_thread = UpdaterThread.UpdaterThread(self.queue, self.server_thread_lock,
                                                          self.T, self.current_t, self.server_network,
                                                          self.async_client_manager, self.stop_event,
                                                          self.test_data, server_config["updater"])
        self.check_in_thread = CheckInThread.CheckInThread(server_config["checkin"], self.async_client_manager,
                                                           self.current_t, self.T)

    def run(self):
        print("Start server:")

        # 启动server中的两个线程
        self.scheduler_thread.start()
        self.updater_thread.start()
        self.check_in_thread.start()

        client_thread_list = self.async_client_manager.get_checked_in_client_thread_list()
        for client_thread in client_thread_list:
            client_thread.join()
        self.scheduler_thread.join()
        print("scheduler_thread joined")
        self.updater_thread.join()
        print("updater_thread joined")
        _async_raise(self.check_in_thread.ident, SystemExit)
        print("check_in_thread joined")

        print("Thread count =", threading.active_count())
        print(*threading.enumerate(), sep="\n")

        if not self.queue.empty():
            print("\nUn-used client weights:", self.queue.qsize())
            for q in range(self.queue.qsize()):
                self.queue.get()
        self.queue.close()

        self.accuracy_list = self.updater_thread.get_accuracy_list()
        del self.scheduler_thread
        del self.updater_thread
        del self.async_client_manager
        del self.check_in_thread
        print("End!")

    def get_accuracy_list(self):
        return self.accuracy_list
