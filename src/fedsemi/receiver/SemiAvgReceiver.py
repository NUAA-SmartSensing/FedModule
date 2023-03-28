class SemiAvgReceiver:
    def __init__(self, queue_list, config):
        self.queue_list = queue_list
        self.config = config
        self.len = len(queue_list)

    def receive(self, nums=1):
        # 第i组/层全都上传完成
        while True:
            for i in range(self.len):
                if self.queue_list[i].qsize() == nums[i]:
                    return i

    def update_queue_len(self):
        self.len = len(self.queue_list)
