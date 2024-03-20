import time

from client.NormalClient import NormalClient


class TWAFLClient(NormalClient):
    def __init__(self, c_id, stop_event, selected_event, delay, train_ds, index_list, config, dev):
        NormalClient.__init__(self, c_id, stop_event, selected_event, delay, train_ds, index_list, config, dev)
        self.flag = False

    def local_task(self):
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
