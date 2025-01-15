import os
import threading

import torch.cuda
from torch.multiprocessing import Event

from core.MessageQueue import DataGetter, MessageQueueFactory
from utils import Time
from utils.GlobalVarGetter import GlobalVarGetter
from utils.ModuleFindTool import load_model_from_config


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

        # 全局模型
        self.train_ds = self.message_queue.get_train_dataset()
        self.model = load_model_from_config(self.server_config.get('model'), self)
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.dev)
        self.global_var['global_model'] = self.model

        # 对模型非更新参数进行检测
        training_params = {k: False for k in self.model.state_dict()}
        for n, p in self.model.named_parameters():
            training_params[n] = p.requires_grad
        self.global_var['training_params'] = training_params

        # 计时变量
        self.T = self.server_config["epochs"]
        self.current_t = Time.Time(1)
        self.schedule_t = Time.Time(1)
        self.global_var['current_t'] = self.current_t
        self.global_var['schedule_t'] = self.schedule_t
        self.global_var['T'] = self.T

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
        # save model
        if "save_model" in self.server_config and self.server_config["save_model"]:
            torch.save(self.model.state_dict(), os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results/", self.global_config["experiment"], "model.pth"))
        # 结束主类
        self.kill_main_class()
        print("End!")

    def get_config(self):
        return self.config

    def kill_main_class(self):
        del self.scheduler_thread
        del self.updater_thread
        del self.queue_manager
        del self.data_getter_thread
