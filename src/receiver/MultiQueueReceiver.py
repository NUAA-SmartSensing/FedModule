import time

from receiver.AbstractReceiver import AbstractReceiver


class MultiQueueReceiver(AbstractReceiver):
    def __init__(self, config):
        super().__init__(config)

    def receive(self, queue, nums):
        # 第i组/层全都上传完成
        while True:
            for i in range(len(nums)):
                if nums[i] == 0:
                    continue
                if queue[i].qsize() == nums[i]:
                    return i
            time.sleep(0.1)
