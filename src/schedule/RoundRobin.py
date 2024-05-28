from schedule.AbstractSchedule import AbstractSchedule


class RoundRobin(AbstractSchedule):
    def __init__(self, config):
        super().__init__(config)
        self.pos = 0
        self.c_ratio = config["c_ratio"]

    def schedule(self, client_list):
        total = len(client_list)
        select_num = int(self.c_ratio * len(client_list))

        print("Current clients:", total, ", select:", select_num)
        if self.pos + select_num <= total:
            selected_client_threads = client_list[self.pos:self.pos + select_num]
        else:
            selected_client_threads = client_list[self.pos:] + client_list[:select_num + self.pos - total]
        self.pos = (self.pos + select_num) % total
        return selected_client_threads
