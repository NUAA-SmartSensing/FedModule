from numgenerator.NumGeneratorFactory import NumGeneratorFactory
from updater.SyncUpdater import SyncUpdater


class AsyncUpdater(SyncUpdater):
    def __init__(self, server_thread_lock, stop_event, config, mutex_sem, empty_sem, full_sem):
        SyncUpdater.__init__(self, server_thread_lock, stop_event, config, mutex_sem, empty_sem, full_sem)
        # number of updates per aggregation.
        self.num_generator = NumGeneratorFactory(self.config['num_generator']).create_num_generator()
        self.nums = self.num_generator.init()

    def get_update_list(self):
        self.nums = self.num_generator.get_num()
        self.queue_manager.receive(self.nums)
        update_list = []
        for i in range(self.nums):
            update_list.append(self.queue_manager.get())
            c_id = update_list[i]["client_id"]
            time_stamp = update_list[i]["time_stamp"]
            self.sum_delay += (self.current_time.get_time() - time_stamp)
            print("Updater received data from client", c_id, "| staleness =", time_stamp, "-",
                  self.current_time.get_time(), "| queue size = ", self.queue_manager.size())
        return update_list
