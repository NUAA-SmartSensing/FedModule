import threading
from server.BaseServer import BaseServer
from utils import ModuleFindTool


class SemiAsyncServer(BaseServer):
    r"""
        semi_async_server adds a new module named group_manager compared with normal_server
    """

    def __init__(self, config):
        BaseServer.__init__(self, config)
        self.group_manager_config = config['group_manager']

        self.mutex_sem = threading.Semaphore(1)
        self.empty_sem = threading.Semaphore(1)
        self.full_sem = threading.Semaphore(0)

        # initialization of the server
        # the process has an order
        queue_manager_class = ModuleFindTool.find_class_by_path(self.queue_manager_config['path'])
        self.queue_manager = queue_manager_class(self.queue_manager_config)
        self.global_var['queue_manager'] = self.queue_manager

        group_manager_class = ModuleFindTool.find_class_by_path(self.group_manager_config['path'])
        self.group_manager = group_manager_class(self.group_manager_config)
        self.global_var['group_manager'] = self.group_manager

        scheduler_class = ModuleFindTool.find_class_by_path(self.server_config['scheduler']['path'])
        self.scheduler = scheduler_class(self.server_thread_lock,
                                         self.server_config['scheduler'],
                                         self.mutex_sem, self.empty_sem, self.full_sem)
        self.global_var['scheduler'] = self.scheduler
        self.scheduler_thread = threading.Thread(target=self.scheduler.run)

        updater_class = ModuleFindTool.find_class_by_path(self.server_config['updater']['path'])
        self.updater = updater_class(self.server_thread_lock, self.server_config['updater'],
                                     self.mutex_sem, self.empty_sem, self.full_sem)
        self.global_var['updater'] = self.updater
        self.updater_thread = threading.Thread(target=self.updater.run)

    def kill_main_class(self):
        del self.scheduler_thread
        del self.updater_thread
        del self.queue_manager
        del self.group_manager
        self.mutex_sem.release()
        self.empty_sem.release()
        self.full_sem.release()
