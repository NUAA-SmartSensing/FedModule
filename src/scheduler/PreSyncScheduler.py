import time

from scheduler.SyncScheduler import SyncScheduler


class PreSyncScheduler(SyncScheduler):
    def __init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem):
        SyncScheduler.__init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem)

    def run(self):
        # 预训练阶段
        print("start pretraining")
        client_list = self.client_manager.get_client_id_list()
        for client_id in client_list:
            # 将server的模型参数和时间戳发给client
            self.send_weights(client_id, -1, 0)
            # 启动一次client线程
            self.client_manager.selected_event_list[client_id].set()
        time.sleep(2)
        while True:
            flag = False
            training_params = self.message_queue.get_training_status()
            for _, v in training_params.items():
                if v is True:
                    flag = True
            if not flag:
                break
            time.sleep(1)
        print("\n-----------------------------------------------------------------pretraining over")

        super().run()
