import time


class SyncReceiver:
    def __init__(self, queue, config):
        self.queue = queue
        self.config = config

    def receive(self, nums=1):
        while self.queue.qsize() < nums:
            time.sleep(0.01)
