import time

from client import NormalClient


class ActiveClient(NormalClient.NormalClient):
    def __init__(self, c_id, init_lock, stop_event, selected_event, delay, index_list, config, dev):
        NormalClient.NormalClient.__init__(self, c_id, init_lock, stop_event, selected_event, delay, index_list, config, dev)
        self.acquire_model_delay = config['acquire_model_delay']

    def run(self):
        self.init_client()
        self.wait_notify()
        while not self.stop_event.is_set():
            # 该client被选中，开始执行本地训练
            if self.event.is_set():
                # 该client进行训练
                data_sum, weights = self.train()
                # client传回server的信息具有延迟
                time.sleep(self.delay)
                print("Client", self.client_id, "trained")

                # 返回其ID、模型参数和时间戳
                self.upload(data_sum, weights)
                time.sleep(self.delay)

                # 获取服务器最新模型
                time.sleep(self.acquire_model_delay)
                while True:
                    latest_model, time_stamp = self.message_queue.get_latest_model()
                    if latest_model is not None:
                        break
                latest_model, time_stamp = self.message_queue.get_latest_model()
                self.model.load_state_dict(latest_model)
                self.time_stamp = time_stamp
            # 该client等待被选中
            else:
                self.event.wait()
