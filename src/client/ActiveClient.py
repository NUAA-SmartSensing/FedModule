import time

from client import NormalClient


class ActiveClient(NormalClient.NormalClient):
    def __init__(self, c_id, queue_manager, stop_event, delay, train_ds, client_config, dev, print_lock, global_var):
        NormalClient.NormalClient.__init__(self, c_id, queue_manager, stop_event, delay, train_ds, client_config, dev,
                                           print_lock, global_var)
        self.acquire_model_delay = client_config['acquire_model_delay']

    def run(self):
        while not self.stop_event.is_set():
            # 初始化
            if self.received_weights:
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
                data_sum, weights = self.train_one_epoch()
                # client传回server的信息具有延迟
                time.sleep(self.delay)
                self.print_lock.acquire()
                print("Client", self.client_id, "trained")
                self.print_lock.release()

                # 返回其ID、模型参数和时间戳
                update_dict = {"client_id": self.client_id, "weights": weights, "data_sum": data_sum,
                               "time_stamp": self.time_stamp}
                self.queue_manager.put(update_dict)

                # 获取服务器最新模型
                time.sleep(self.acquire_model_delay)
                self.global_var['scheduler'].server_thread_lock.acquire()
                self.model.load_state_dict(self.global_var['scheduler'].server_weights)
                self.time_stamp = self.global_var['scheduler'].current_t.get_time()
                self.global_var['scheduler'].server_thread_lock.release()
                time.sleep(self.acquire_model_delay)
                self.client_thread_lock.release()
            # 该client等待被选中
            else:
                self.event.wait()
