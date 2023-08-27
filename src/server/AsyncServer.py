from server import BaseServer
from utils import ModuleFindTool


class AsyncServer(BaseServer.BaseServer):
    def __init__(self, config):
        BaseServer.BaseServer.__init__(self, config)

        # initialization of the server
        # the process has an order
        queue_manager_class = ModuleFindTool.find_class_by_path(self.queue_manager_config['path'])
        self.queue_manager = queue_manager_class(self.queue_manager_config)
        self.global_var['queue_manager'] = self.queue_manager

        client_manager_class = ModuleFindTool.find_class_by_path(self.client_manager_config['path'])
        self.client_manager = client_manager_class(self.stop_event, self.client_manager_config)
        self.global_var['client_manager'] = self.client_manager

        # client_manager初始化
        self.client_manager.start_all_clients()

        scheduler_class = ModuleFindTool.find_class_by_path(self.server_config['scheduler']['path'])
        self.scheduler_thread = scheduler_class(self.server_thread_lock, self.server_config["scheduler"])
        self.global_var['scheduler'] = self.scheduler_thread

        updater_class = ModuleFindTool.find_class_by_path(self.server_config['updater']['path'])
        self.updater_thread = updater_class(self.server_thread_lock, self.stop_event, self.server_config['updater'])
        self.global_var['updater'] = self.updater_thread
