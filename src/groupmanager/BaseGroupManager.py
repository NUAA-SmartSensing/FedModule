from utils.GlobalVarGetter import GlobalVarGetter


class BaseGroupManager:
    def __init__(self, config):
        self.config = config
        self.global_var = GlobalVarGetter.get()

    def __group(self, *args, **kwargs):
        pass

    def get_group_num(self, *args, **kwargs):
        pass

    def get_group_list(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass
