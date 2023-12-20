import time

from scheduler.SyncScheduler import SyncScheduler


class PreSyncScheduler(SyncScheduler):
    def __init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem):
        SyncScheduler.__init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem)

    def run(self):
        # 预训练阶段
        print("start pretraining")
        self.empty_sem.acquire()
        self.mutex_sem.acquire()
        client_list = self.client_manager.get_client_id_list()
        for client_id in client_list:
            # 将server的模型参数和时间戳发给client
            self.send_weights(client_id, -1, 0)
            # 启动一次client线程
            self.client_manager.selected_event_list[client_id].set()
        self.queue_manager.receive(len(client_list))
        # 通知updater聚合权重
        self.mutex_sem.release()
        self.full_sem.release()
        time.sleep(0.01)

        print("\n-----------------------------------------------------------------pretraining over")

        super().run()
