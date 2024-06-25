from group.OneGroup import OneGroup


class GroupCaller:
    def __init__(self, group_manager):
        self.group_manager = group_manager

    def group(self, client_list, latency_list, *args, **kwargs):
        if isinstance(self.group_manager.group_method, OneGroup):
            return self.group_manager.group_method.group(client_list)
        else:
            return self.group_manager.group_method.group(client_list, latency_list, *args, **kwargs)

    def check_update(self, *args, **kwargs):
        return self.group_manager.group_method.check_update()
