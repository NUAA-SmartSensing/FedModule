import threading

from server import BaseServer
from utils import ModuleFindTool


class LeoSyncServer(BaseServer.BaseServer):
    def __init__(self, config):
        BaseServer.BaseServer.__init__(self, config)

        self.mutex_sem = threading.Semaphore(1)
        self.empty_sem = threading.Semaphore(1)
        self.full_sem = threading.Semaphore(0)

        # initialization of the server
        # the process has an order
        queue_manager_class = ModuleFindTool.find_class_by_path(self.queue_manager_config['path'])
        self.queue_manager = queue_manager_class(self.queue_manager_config)
        self.global_var['queue_manager'] = self.queue_manager

        if 'group_manager' in config.keys():
            #创建group_manager
            self.group_manager_config = config['group_manager']
            group_manager_class = ModuleFindTool.find_class_by_path(self.group_manager_config['path'])
            self.group_manager = group_manager_class(self.group_manager_config)
            self.global_var['group_manager'] = self.group_manager

        scheduler_class = ModuleFindTool.find_class_by_path(self.server_config['scheduler']['path'])
        self.scheduler_thread = scheduler_class(self.server_thread_lock, self.server_config["scheduler"],
                                                self.mutex_sem, self.empty_sem, self.full_sem)
        self.global_var['scheduler'] = self.scheduler_thread

        updater_class = ModuleFindTool.find_class_by_path(self.server_config['updater']['path'])
        self.updater_thread = updater_class(self.server_thread_lock, self.stop_event, self.server_config['updater'],
                                            self.mutex_sem, self.empty_sem, self.full_sem)
        self.global_var['updater'] = self.updater_thread

    def kill_main_class(self):
        del self.scheduler_thread
        del self.updater_thread
        del self.queue_manager
        self.mutex_sem.release()
        self.empty_sem.release()
        self.full_sem.release()
