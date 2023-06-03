from numgenerator.AbstractNumGenerator import AbstractNumGenerator


class StaticNumGenerator(AbstractNumGenerator):
    def __init__(self, nums):
        self.nums = nums

    def init(self, *args, **kwargs):
        return self.nums

    def get_num(self, *args, **kwargs):
        return self.nums
