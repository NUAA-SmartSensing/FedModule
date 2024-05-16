import time

from receiver.AbstractReceiver import AbstractReceiver


class NormalReceiver(AbstractReceiver):
    def __init__(self, config):
        super().__init__(config)

    # to support any queue_manger
    def receive(self, queue, nums):
        while queue.qsize() < nums:
            time.sleep(0.01)
