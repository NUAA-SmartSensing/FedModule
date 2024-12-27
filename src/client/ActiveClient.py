import time

from client.NormalClient import NormalClient


class ActiveClient(NormalClient):
    def __init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev):
        NormalClient.__init__(self, c_id, stop_event, selected_event, delay, index_list, config, dev)
        self.acquire_model_delay = config['acquire_model_delay']

    def _run_iteration(self):
        if not self.event.is_set():
            self.event.wait()
        while not self.stop_event.is_set():
            self.local_run()
            time.sleep(0.01)

    def delay_simulate(self):
        time.sleep(self.acquire_model_delay + self.delay)
