import time


class MultiQueueReceiver:
    def __init__(self, config):
        self.config = config

    def receive(self, queue, nums):
        # 第i组/层全都上传完成
        while True:
            for i in range(len(nums)): # len(nums)即为组的数量
                if queue[i].qsize() == nums[i]: # 即queue收到的消息数量与选中该group下的schedule的client数量相同，即全部收齐了
                    return i # 返回第几组
            time.sleep(0.1)
