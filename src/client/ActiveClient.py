import time

from client import NormalClient


class ActiveClient(NormalClient.NormalClient):
    def run(self):
        while not self.stop_event.is_set():
            # 初始化
            if self.received_weights:
                self.model.load_state_dict(self.weights_buffer, strict=True)
                self.event_is_set = False
            if self.received_time_stamp:
                self.time_stamp = self.time_stamp_buffer
                self.received_time_stamp = False
            if self.event_is_set:
                self.event_is_set = False

            # 该client被选中，开始执行本地训练
            if self.event.is_set():
                self.client_thread_lock.acquire()
                # 该client进行训练
                data_sum, weights = self.train_one_epoch()
                # client传回server的信息具有延迟
                time.sleep(self.delay)

                # 返回其ID、模型参数和时间戳
                update_dict = {"client_id": self.client_id, "weights": weights, "data_sum": data_sum,
                               "time_stamp": self.time_stamp}
                self.queue_manager.put(update_dict)
                # 获取服务器最新模型
                self.model.load_state_dict(self.global_var['scheduler'].server_weights)
                self.client_thread_lock.release()
            # 该client等待被选中
            else:
                self.event.wait()
