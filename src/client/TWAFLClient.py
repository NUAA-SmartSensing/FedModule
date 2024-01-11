import time

from client.NormalClient import NormalClient


class TWAFLClient(NormalClient):
    def __init__(self, c_id, stop_event, selected_event, delay, train_ds, index_list, config, dev):
        NormalClient.__init__(self, c_id, stop_event, selected_event, delay, train_ds, index_list, config, dev)
        self.flag = False

    def run(self):
        self.init_client()
        while not self.stop_event.is_set():
            # 该client被选中，开始执行本地训练
            if self.event.is_set():
                self.event.clear()
                self.message_queue.set_training_status(self.client_id, True)
                self.wait_notify()
                self.flag = self.time_stamp % 15 in [11, 13, 14, 12, 0]
                # 该client进行训练
                data_sum, weights = self.train()

                # client传回server的信息具有延迟
                print("Client", self.client_id, "trained")
                if self.flag:
                    time.sleep(self.delay)
                else:
                    time.sleep(self.delay / 2)

                # 返回其ID、模型参数和时间戳
                self.upload(data_sum, weights)

                self.message_queue.set_training_status(self.client_id, False)
            # 该client等待被选中
            else:
                self.event.wait()
