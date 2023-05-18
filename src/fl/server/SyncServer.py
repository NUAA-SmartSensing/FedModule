import threading

from fl.server import BaseServer
from fedsync import SchedulerThread
from fedsync import SyncClientManager
from fedsync import UpdaterThread


class SyncServer(BaseServer.BaseServer):
    def __init__(self, config, global_config, server_config, client_config, manager_config):
        BaseServer.BaseServer.__init__(config, global_config, server_config)
        init_weights = self.server_network.state_dict()
        datasets = self.dataset.get_train_dataset()

        self.mutex_sem = threading.Semaphore(1)
        self.empty_sem = threading.Semaphore(1)
        self.full_sem = threading.Semaphore(0)
        self.client_manager = SyncClientManager.SyncClientManager(init_weights, global_config["client_num"],
                                                                  global_config["multi_gpu"],
                                                                  datasets, self.queue, self.current_t, self.schedule_t,
                                                                  self.stop_event, client_config, manager_config,
                                                                  self.global_var)
        self.global_var['client_manager'] = self.client_manager
        self.scheduler_thread = SchedulerThread.SchedulerThread(self.server_thread_lock, self.client_manager,
                                                                self.queue, self.current_t, self.schedule_t,
                                                                server_config["scheduler"],
                                                                self.server_network, self.T,
                                                                self.mutex_sem, self.empty_sem, self.full_sem,
                                                                self.global_var)
        self.global_var['scheduler'] = self.scheduler_thread
        self.updater_thread = UpdaterThread.UpdaterThread(self.queue, self.server_thread_lock,
                                                          self.T, self.current_t, self.schedule_t, self.server_network,
                                                          self.stop_event,
                                                          self.test_data, server_config["updater"],
                                                          self.mutex_sem, self.empty_sem, self.full_sem,
                                                          self.global_var)
        self.global_var['updater'] = self.updater_thread
