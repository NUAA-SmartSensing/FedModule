import time


class AvgReceiver:
    def __init__(self, queue):
        self.queue = queue

    def start(self, nums):
        while self.queue.qsize() < nums:
            time.sleep(0.01)
