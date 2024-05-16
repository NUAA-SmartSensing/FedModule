import time

from scheduler.BaseScheduler import BaseScheduler


class SyncScheduler(BaseScheduler):
    def __init__(self, server_thread_lock, config, mutex_sem, empty_sem, full_sem):
        BaseScheduler.__init__(self, server_thread_lock, config)
        self.mutex_sem = mutex_sem
        self.empty_sem = empty_sem
        self.full_sem = full_sem

    def run(self):
        while True:
            # 每隔一段时间进行一次schedule
            self.empty_sem.acquire()
            self.mutex_sem.acquire()
            current_time = self.current_t.get_time()
            schedule_time = self.schedule_t.get_time()
            if current_time > self.T:
                break
            nums = self.schedule(current_time, schedule_time)
            self.schedule_t.time_add()
            # 等待所有客户端上传更新
            self.queue_manager.receive(nums)

            # 通知updater聚合权重
            self.mutex_sem.release()
            self.full_sem.release()
            time.sleep(0.01)

    def schedule(self, current_time, schedule_time):
        r"""
            schedule the clients

            Args:
                current_time: the number of aggregation
                schedule_time: the number scheduling clients

            Returns:
                the number of clients
        """
        selected_client = self.client_select()
        print(f"| current_epoch {current_time} |. Begin client select")
        print("\nSchedulerThread select(", len(selected_client), "clients):")
        for client_id in selected_client:
            print(client_id, end=" | ")
            # 将server的模型参数和时间戳发给client
            self.send_weights(client_id, current_time, schedule_time)
            # 启动一次client线程
            self.selected_event_list[client_id].set()
        print("\n-----------------------------------------------------------------Schedule complete")
        return len(selected_client)