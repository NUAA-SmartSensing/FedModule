import time

from updater.BaseUpdater import BaseUpdater


class SyncUpdater(BaseUpdater):
    def __init__(self, server_thread_lock, stop_event, config, mutex_sem, empty_sem, full_sem):
        BaseUpdater.__init__(self, server_thread_lock, stop_event, config)
        self.mutex_sem = mutex_sem
        self.empty_sem = empty_sem
        self.full_sem = full_sem

    def run(self):
        for epoch in range(self.T):
            self.full_sem.acquire()
            self.mutex_sem.acquire()
            update_list = []
            # 接收所有的更新
            while not self.queue_manager.empty():
                update_list.append(self.queue_manager.get())

            self.server_thread_lock.acquire()
            self.update_server_weights(epoch, update_list)
            self.run_server_test(epoch)
            self.server_thread_lock.release()

            self.current_time.time_add()
            # 本轮结束
            self.mutex_sem.release()
            self.empty_sem.release()
            time.sleep(0.01)

        # 终止所有client线程
        self.client_manager.stop_all_clients()

