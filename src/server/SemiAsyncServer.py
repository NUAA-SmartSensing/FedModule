import threading
from server import BaseServer
from utils import ModuleFindTool


class SemiAsyncServer(BaseServer.BaseServer):
    def __init__(self, config):
        BaseServer.BaseServer.__init__(self, config)
        self.group_manager_config = config['group_manager']

        self.mutex_sem = threading.Semaphore(1)
        self.empty_sem = threading.Semaphore(1)
        self.full_sem = threading.Semaphore(0)

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

        group_manager_class = ModuleFindTool.find_class_by_path(self.group_manager_config['path'])
        self.group_manager = group_manager_class(self.group_manager_config)
        self.global_var['group_manager'] = self.group_manager

        updater_class = ModuleFindTool.find_class_by_path(self.server_config['updater']['path'])
        self.updater_thread = updater_class(self.server_thread_lock, self.stop_event, self.server_config['updater'],
                                            self.mutex_sem, self.empty_sem, self.full_sem)
        self.global_var['updater'] = self.updater_thread

        scheduler_class = ModuleFindTool.find_class_by_path(self.server_config['scheduler']['path'])
        self.scheduler_thread = scheduler_class(self.server_thread_lock,
                                                self.server_config['scheduler'],
                                                self.mutex_sem, self.empty_sem, self.full_sem)
        self.global_var['scheduler'] = self.scheduler_thread

    def kill_main_class(self):
        del self.scheduler_thread
        del self.updater_thread
        del self.client_manager
        del self.queue_manager
        del self.group_manager
        self.mutex_sem.release()
        self.empty_sem.release()
        self.full_sem.release()
