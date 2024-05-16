import time

from client.NormalClient import NormalClient


class ActiveClient(NormalClient):
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        NormalClient.__init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.group_id = None
        self.acquire_model_delay = config['acquire_model_delay']

    def run(self):
        """
        The primary running function of Client is used for clients with a base class of process,
        which executes before being woken up by the server.
        """
        self.init_client()
        while not self.stop_event.is_set():
            # 该client被选中，开始执行本地训练
            if self.event.is_set():
                self.local_run()
            # 该client等待被选中
            else:
                self.event.wait()

    def local_run(self):
        """
        The run function of Client runs the main body, suitable for use as a target parameter of process.
        """
        self.message_queue.set_training_status(self.client_id, True)
        self.local_task()
        # 获取服务器最新模型
        self.delay_simulate(self.acquire_model_delay)
        while True:
            if self.stop_event.is_set():
                break
            latest_model, time_stamp = self.message_queue.get_latest_model()
            if latest_model is not None and time_stamp != self.time_stamp:
                break
            else:
                time.sleep(0.01)
        latest_model, time_stamp = self.message_queue.get_latest_model()
        self.model.load_state_dict(latest_model)
        self.time_stamp = time_stamp
        self.message_queue.set_training_status(self.client_id, False)

