import os
import threading

import torch.cuda
from torch.multiprocessing import Event

from utils import ModuleFindTool, Time
from utils.DatasetUtils import FLDataset
from utils.GlobalVarGetter import GlobalVarGetter
from core.MessageQueue import DataGetter, MessageQueueFactory


class BaseServer:
    def __init__(self, config):
        self.config = config
        self.global_config = config['global']
        self.server_config = config['server']
        self.client_config = config['client']
        self.queue_manager_config = config['queue_manager']

        # 消息队列
        self.message_queue = MessageQueueFactory.create_message_queue()
        # 全局存储变量
        self.global_var = GlobalVarGetter.get()
        self.global_var['server'] = self

        # 数据集
        self.train_ds = self.message_queue.get_train_dataset()
        self.fl_train_ds = FLDataset(self.train_ds, range(len(self.train_ds)))

        # 全局模型
        if isinstance(self.server_config["model"], dict):
            model_class = ModuleFindTool.find_class_by_path(self.server_config["model"]["path"])
            for k, v in self.server_config["model"]["params"].items():
                if isinstance(v, str):
                    self.server_config["model"]["params"][k] = eval(v)
            self.server_network = model_class(**self.server_config["model"]["params"])
        elif isinstance(self.server_config["model"], str):
            self.server_network = torch.load(self.server_config["model"])
        else:
            raise ValueError("model config error")
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.server_network = self.server_network.to(self.dev)
        self.global_var['server_network'] = self.server_network

        # 对模型非更新参数进行检测
        d = self.server_network.state_dict()
        w = list(self.server_network.parameters())
        i = 0
        training_params = {}
        for k in d:
            try:
                if isinstance(w[i], type(d[k])) and d[k].equal(w[i]):
                    i += 1
                    training_params[k] = True
                else:
                    training_params[k] = False
            except:
                training_params[k] = False
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
        # process event
        self.stop_event = Event()
        self.stop_event.clear()
        self.server_thread_lock = threading.Lock()

        # 主类
        # initialization of the server
        # the process has an order
        self.queue_manager = None
        self.scheduler_thread = None
        self.updater_thread = None
        self.data_getter_thread = DataGetter()

    def run(self):
        print("Start server:")

        # 启动server中的三个线程
        self.data_getter_thread.start()
        self.scheduler_thread.start()
        self.updater_thread.start()

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
        # save model
        if "save_model" in self.server_config and self.server_config["save_model"]:
            torch.save(self.server_network.state_dict(), os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/", self.global_config["experiment"], "model.pth"))
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
        del self.queue_manager
        del self.data_getter_thread
