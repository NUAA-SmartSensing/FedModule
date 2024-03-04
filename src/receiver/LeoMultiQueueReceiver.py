import time

class LeoMultiQueueReceiver:
    def __init__(self, config):
        self.config = config

    def receive(self, queue, nums, group_id_next):
        ready_groups_list = [0 for _ in range(len(nums))]
        # 直到需要调度的group完成训练
        while True:
            for i in range(len(nums)): # len(nums)即为组的数量
                if queue[i].qsize() == nums[i]: # 即queue收到的消息数量与选中该group下的schedule的client数量相同，即全部收齐了
                    ready_groups_list[i] = 1 #表示某组准备就绪
            if ready_groups_list[group_id_next] != 1:
                time.sleep(0.1)
            else:
                return ready_groups_list





        # ready_groups_list = [0 for _ in range(len(nums))]
        # 第i组/层全都上传完成
        # while True:
        #     for i in range(len(nums)): # len(nums)即为组的数量
        #         if queue[i].qsize() == nums[i]: # 即queue收到的消息数量与选中该group下的schedule的client数量相同，即全部收齐了
        #             ready_groups_list[i] = 1 #表示某组准备就绪
        #             return i # 返回第几组
        #     time.sleep(0.1)
