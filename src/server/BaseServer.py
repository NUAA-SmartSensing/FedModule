import threading

import torch.cuda
from torch.multiprocessing import Event
from torch.utils.data import DataLoader

from utils import ModuleFindTool, Time
from utils.DataReader import CustomDataset
from utils.GlobalVarGetter import GlobalVarGetter
from utils.ProcessManager import DataGetter, MessageQueueFactory


def _read_data(dataset):
    data = []
    targets = []
    dl = DataLoader(dataset, batch_size=1)
    for x, y in dl:
        data.append(x[0])
        targets.append(y[0])
    data = torch.stack(data)
    targets = torch.stack(targets)
    data.share_memory_()
    targets.share_memory_()
    return data, targets


class BaseServer:
    def __init__(self, config):
        self.config = config
        self.global_config = config['global']
        self.server_config = config['server']
        self.client_config = config['client']
        self.client_manager_config = config['client_manager']
        self.queue_manager_config = config['queue_manager']

        # 消息队列
        self.message_queue = MessageQueueFactory.create_message_queue()
        # 全局存储变量
        self.global_var = GlobalVarGetter().get()
        self.global_var['server'] = self

        # 数据集
        dataset_class = ModuleFindTool.find_class_by_path(self.global_config["dataset"]["path"])
        self.dataset = dataset_class(self.global_config["client_num"], self.global_config["iid"],
                                     self.global_config["dataset"]["params"])
        self.global_var['dataset'] = self.dataset
        self.config['global']['iid'] = self.dataset.get_config()
        # 根据配置情况将数据集发送给客户端
        self.send_dataset()

        # 全局模型
        model_class = ModuleFindTool.find_class_by_path(self.server_config["model"]["path"])
        for k, v in self.server_config["model"]["params"].items():
            if isinstance(v, str):
                self.server_config["model"]["params"][k] = eval(v)
        self.server_network = model_class(**self.server_config["model"]["params"])
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.server_network = self.server_network.to(self.dev)
        self.global_var['server_network'] = self.server_network

        # 对模型非更新参数进行检测
        d = self.server_network.state_dict()
        w = [v for v in self.server_network.parameters()]
        i = 0
        training_params = {}
        for k in d:
            if not d[k].equal(w[i]):
                training_params[k] = False
            else:
                i += 1
                training_params[k] = True

        self.message_queue.set_training_params(training_params)

        # 计时变量
        self.T = self.server_config["epochs"]
        self.current_t = Time.Time(1)
        self.schedule_t = Time.Time(1)
        self.global_var['current_t'] = self.current_t
        self.global_var['schedule_t'] = self.schedule_t
        self.global_var['T'] = self.T

        # 运行时变量
        self.accuracy_list = []
        self.loss_list = []
        self.print_lock = threading.Lock()
        self.global_var['print_lock'] = self.print_lock
        # process event
        self.stop_event = Event()
        self.stop_event.clear()
        self.server_thread_lock = threading.Lock()

        # 主类
        # initialization of the server
        # the process has an order
        self.queue_manager = None
        self.client_manager = None
        self.scheduler_thread = None
        self.updater_thread = None
        self.data_getter_thread = DataGetter()

    def run(self):
        print("Start server:")

        # 启动server中的三个线程
        self.data_getter_thread.start()
        self.scheduler_thread.start()
        self.updater_thread.start()

        client_list = self.client_manager.get_client_list()
        for client in client_list:
            client.join()
        self.scheduler_thread.join()
        print("scheduler_thread joined")
        self.updater_thread.join()
        print("updater_thread joined")
        self.data_getter_thread.kill()
        self.data_getter_thread.join()
        print("data_getter_thread joined")

        # 队列报告
        self.queue_manager.stop()

        self.accuracy_list, self.loss_list = self.updater_thread.get_accuracy_and_loss_list()
        # 结束主类
        self.kill_main_class()
        print("End!")

    def get_accuracy_and_loss_list(self):
        return self.accuracy_list, self.loss_list

    def get_config(self):
        return self.config

    def kill_main_class(self):
        del self.scheduler_thread
        del self.updater_thread
        del self.client_manager
        del self.queue_manager
        del self.data_getter_thread

    def send_dataset(self):
        # 预加载
        if 'dataset_pre_load' in self.global_config and self.global_config['dataset_pre_load']:
            dataset = self.dataset.get_train_dataset()
            data, targets = _read_data(dataset)
            self.message_queue.set_dataset(CustomDataset(data, targets))
            self.dataset.delete_train_dataset()
        # 静态加载
        else:
            self.message_queue.set_dataset(self.dataset.get_train_dataset())
