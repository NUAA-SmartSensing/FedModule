from numgenerator.NumGeneratorFactory import NumGeneratorFactory
from updater.BaseUpdater import BaseUpdater


class AsyncUpdater(BaseUpdater):
    def __init__(self, server_thread_lock, stop_event, config, mutex_sem, empty_sem, full_sem):
        BaseUpdater.__init__(self, server_thread_lock, stop_event, config)
        self.mutex_sem = mutex_sem
        self.empty_sem = empty_sem
        self.full_sem = full_sem
        # number of updates per aggregation.
        self.num_generator = NumGeneratorFactory(self.config['num_generator']).create_num_generator()
        self.nums = self.num_generator.init()

    def run(self):
        for epoch in range(self.T):
            self.full_sem.acquire()
            self.mutex_sem.acquire()

            update_list = self.get_update_list()
            # Update the global model using the received updates from the clients.
            self.server_thread_lock.acquire()
            self.server_update(epoch, update_list)
            self.server_thread_lock.release()

            self.current_time.time_add()
            self.mutex_sem.release()
            self.empty_sem.release()

        print("Average delay =", (self.sum_delay / self.T))

    def server_update(self, epoch, update_list):
        self.update_server_weights(epoch, update_list)
        self.run_server_test(epoch)

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
