import time

import torch

from client import TestClient


class DLClient(TestClient.TestClient):
    def __init__(self, c_id, stop_event, delay, train_ds, config, dev):
        TestClient.TestClient.__init__(self, c_id, stop_event, delay, train_ds, config, dev)
        self.init = False

    def run(self):
        while not self.stop_event.is_set():
            if self.received_weights:
                # 更新模型参数
                if self.init:
                    if self.client_id in self.weights_buffer.keys():
                        for key, var in self.model.state_dict().items():
                            if torch.cuda.is_available():
                                self.weights_buffer[self.client_id][key] = self.weights_buffer[self.client_id][key].to(
                                    self.dev)
                            self.weights_buffer[self.client_id][key] = self.config['alpha'] * var + (
                                        1 - self.config['alpha']) * self.weights_buffer[self.client_id][key]
                        self.model.load_state_dict(self.weights_buffer[self.client_id], strict=True)
                else:
                    self.model.load_state_dict(self.weights_buffer['global'], strict=True)
                    self.init = True
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
                # 本地测试
                self.run_test()
                time.sleep(self.delay)

                # 返回其ID、模型参数和时间戳
                update_dict = {"client_id": self.client_id, "weights": weights, "data_sum": data_sum,
                               "time_stamp": self.time_stamp}
                self.queue_manager.put(update_dict)
                self.event.clear()
                self.client_thread_lock.release()
            # 该client等待被选中
            else:
                self.event.wait()
