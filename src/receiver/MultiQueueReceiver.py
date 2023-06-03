class MultiQueueReceiver:
    def __init__(self, config):
        self.config = config

    def receive(self, queue, nums):
        # 第i组/层全都上传完成
        while True:
            for i in range(len(nums)):
                if queue[i].qsize() == nums[i]:
                    return i
