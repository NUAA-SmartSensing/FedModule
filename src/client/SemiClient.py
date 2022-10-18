import copy
import time

from client import SyncClient


class SemiClient(SyncClient.SyncClient):
    def __init__(self, c_id, queue_manager, stop_event, delay, train_ds, client_config, dev):
        super().__init__(c_id, queue_manager, stop_event, delay, train_ds, client_config, dev)
        self.group_id = 0

    def run(self):
        while not self.stop_event.is_set():
            if self.received_weights:
                # 更新模型参数
                self.model.load_state_dict(self.weights_buffer, strict=True)
                self.received_weights = False
            if self.received_time_stamp:
                self.time_stamp = self.time_stamp_buffer
                self.received_time_stamp = False
            if self.event_is_set:
                self.event_is_set = False

            # 该client被选中，开始执行本地训练
            if self.event.is_set():
                self.client_thread_lock.acquire()
                # 该client进行训练
                r_weights = copy.deepcopy(self.model.state_dict())
                data_sum, weights = self.train_one_epoch(r_weights)

                # client传回server的信息具有延迟
                print("Client", self.client_id, "trained")
                time.sleep(self.delay)

                # 返回其ID、模型参数和时间戳
                update_dict = {"client_id": self.client_id, "weights": weights, "data_sum": data_sum,
                               "time_stamp": self.time_stamp, "group_id": self.group_id}
                self.queue_manager.put(update_dict)
                self.event.clear()
                self.client_thread_lock.release()
            # 该client等待被选中
            else:
                self.event.wait()

    def set_group_id(self, group_id):
        self.group_id = group_id
