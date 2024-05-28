import random

from schedule.AbstractSchedule import AbstractSchedule


class RandomSchedule(AbstractSchedule):
    def __init__(self, config):
        super().__init__(config)
        self.c_ratio = config["c_ratio"]

    def schedule(self, client_list):
        select_num = int(self.c_ratio * len(client_list))

        print("Current clients:", len(client_list), ", select:", select_num)
        selected_client_threads = random.sample(client_list, select_num)
        return selected_client_threads
