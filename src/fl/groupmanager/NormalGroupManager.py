from fl.groupmanager.BaseGroupManager import BaseGroupManager


class NormalGroupManager(BaseGroupManager):
    def __init__(self, config):
        BaseGroupManager.__init__(self, config)

    def group(self, client_list, latency_list, config):
        step = config["step"]
        num = max(latency_list) // step
        tmp_lists = [[] for i in range(num + 1)]
        for i in range(len(latency_list)):
            tmp_lists[latency_list[i] // step].append(client_list[i])

        group_list = []
        for i in tmp_lists:
            if len(i):
                group_list.append(i)
        return group_list, len(group_list)
