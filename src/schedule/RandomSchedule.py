import random


class RandomSchedule:
    def schedule(self, client_list, params):
        select_num = int(params["c_ratio"] * len(client_list))
        if select_num < params["schedule_interval"] + 1:
            select_num = params["schedule_interval"] + 1

        print("Current clients:", len(client_list), ", select:", select_num)
        selected_client_threads = random.sample(client_list, select_num)
        return selected_client_threads
