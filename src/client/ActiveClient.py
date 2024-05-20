import time

from client.NormalClient import NormalClient


class ActiveClient(NormalClient):
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        NormalClient.__init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.group_id = None
        self.acquire_model_delay = config['acquire_model_delay']

    def run(self):
        """
        When it is selected by the server, the client starts local training and never stops until stop event is set.
        """
        self.init_client()
        while not self.stop_event.is_set():
            # The client is selected and starts local training.
            if self.event.is_set():
                self.local_run()
            # The client waits to be selected.
            else:
                self.event.wait()

    def local_run(self):
        """
        The run function of Client runs the main body, suitable for use as a target parameter of process.
        """
        self.message_queue.set_training_status(self.client_id, True)
        self.local_task()
        # get the latest global model
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

