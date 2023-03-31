import time

from client import NormalClient
from utils.ModelTraining import train_one_epoch


class DLClient(NormalClient.NormalClient):
    def __init__(self, c_id, queue_manager, stop_event, delay, train_ds, client_config, dev, global_var):
        NormalClient.NormalClient.__init__(self, c_id, queue_manager, stop_event, delay, train_ds, client_config, dev, global_var)
        self.init = False

    def run(self):
        while not self.stop_event.is_set():
            if self.received_weights:
                # 更新模型参数
                if self.init:
                    for key, var in self.model.state_dict().items():
                        self.weights_buffer[self.client_id][key] = self.config['alpha'] * var + (1 - self.config['alpha']) * self.weights_buffer[self.client_id][key]
                    self.model.load_state_dict(self.weights_buffer[self.client_id], strict=True)
                else:
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
                data_sum, weights = train_one_epoch(self.epoch, self.dev, self.train_dl, self.model, self.loss_func,
                                                    self.opti, self.mu)

                # client传回server的信息具有延迟
                print("Client", self.client_id, "trained")
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
