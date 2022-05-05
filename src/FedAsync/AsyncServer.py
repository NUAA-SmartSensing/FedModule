import threading
import ctypes
import inspect

import torch.cuda

import Queue
from torch import optim
import DataSet.MNIST as MNIST
import Model.CNN as CNN

import AsyncClientManager
import CheckInThread
import SchedulerThread
import UpdaterThread
import Time


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


class AsyncServer:
    def __init__(self, data_type, model_name, c_r, clients_num,
                 batch_size, e, t, schedule_interval, check_in_interval, check_in_num,
                 c_s_list):
        self.data_type = data_type
        self.model_name = model_name
        self.client_ratio = c_r
        self.clients_num = clients_num
        self.batch_size = batch_size
        self.e = e
        self.T = t
        self.schedule_interval = schedule_interval
        self.check_in_interval = check_in_interval
        self.check_in_num = check_in_num
        self.client_staleness_list = c_s_list

        self.current_t = Time.Time(0)
        self.queue = Queue.Queue()
        self.stop_event = threading.Event()
        self.stop_event.clear()

        self.accuracy_and_loss_list = []

        if self.data_type == "MNIST":
            self.dataset = MNIST.MNIST(clients_num, False)
        else:
            self.dataset = MNIST.MNIST(clients_num, False)

        self.test_data = self.dataset.get_test_dataset()
        if self.model_name == "CNN":
            self.server_network = CNN.CNN()
        else:
            self.server_network = CNN.CNN()
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.server_network = self.server_network.to()
        self.opti = optim.Adam(self.server_network.parameters(), lr=0.01)
        init_weights = self.server_network.state_dict()
        datasets = self.dataset.get_train_dataset()
        self.server_thread_lock = threading.Lock()
        self.async_client_manager = AsyncClientManager.AsyncClientManager(init_weights, clients_num, datasets,
                                                                          self.queue, self.batch_size, self.e,
                                                                          self.current_t, self.stop_event,
                                                                          self.client_staleness_list)
        self.scheduler_thread = SchedulerThread.SchedulerThread(self.server_thread_lock, self.schedule_interval,
                                                                self.check_in_interval, self.async_client_manager,
                                                                self.queue, self.client_ratio, self.current_t,
                                                                self.server_network, self.T)
        self.updater_thread = UpdaterThread.UpdaterThread(self.queue, self.server_thread_lock,
                                                          self.T, self.current_t, self.server_network,
                                                          self.async_client_manager, self.stop_event, self.test_data)
        self.check_in_thread = CheckInThread.CheckInThread(self.check_in_interval, self.check_in_num,
                                                           self.async_client_manager, self.current_t, self.T)

    def run(self):
        print("Start server:")

        # 启动server中的两个线程
        self.scheduler_thread.start()
        self.updater_thread.start()
        self.check_in_thread.start()

        # client_thread_list = self.async_client_manager.get_client_thread_list()
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

        self.accuracy_and_loss_list = self.updater_thread.get_accuracy_and_loss_list()
        # self.scheduler_thread.handled = True
        del self.scheduler_thread
        del self.updater_thread
        del self.async_client_manager
        del self.check_in_thread
        print("End!")

    def get_accuracy_and_loss_list(self):
        return self.accuracy_and_loss_list
