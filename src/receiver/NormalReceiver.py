import time


class NormalReceiver:
    def __init__(self, config):
        self.config = config

    # to support any queue_manger
    def receive(self, queue, nums):
        while queue.qsize() < nums:
            time.sleep(0.01)
