from group.AbstractGroup import AbstractGroup


class DelayGroup(AbstractGroup):
    def __init__(self, group_manager, config):
        self.group_manager = group_manager
        self.step = config["step"]
        self.init = False

    def group(self, client_list, latency_list): # latency_list即stale_list
        self.init = True
        num = max(latency_list) // self.step
        tmp_lists = [[] for _ in range(num + 1)]
        for i in range(len(latency_list)):
            tmp_lists[latency_list[i] // self.step].append(client_list[i]) # stale_list 按照 step分多少个区[0,step, k*step],而这些区又对应哪些client_id

        group_list = []
        for i in tmp_lists:
            if len(i):
                group_list.append(i)
        return group_list, len(group_list)

    def check_update(self):
        return self.init
