from time import sleep

from receiver.AbstractReceiver import AbstractReceiver


class VCMReceiver(AbstractReceiver):
    def __init__(self, config):
        super().__init__(config)

    def receive(self, queue_manager, nums):
        while queue_manager.client_num < nums:
            sleep(0.01)
