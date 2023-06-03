from group.AbstractGroup import AbstractGroup


class OneGroup(AbstractGroup):
    def __init__(self, group_manager):
        self.init = False
        self.group_manager = group_manager

    def group(self, client_list):
        self.init = True
        return [client_list], 1

    def check_update(self):
        return self.init
