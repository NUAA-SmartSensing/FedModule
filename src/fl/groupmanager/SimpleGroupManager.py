from fl.groupmanager.BaseGroupManager import BaseGroupManager


class SimpleGroupManager(BaseGroupManager):
    def __init__(self, config):
        BaseGroupManager.__init__(self, config)

    def group(self, client_list, latency_list, config):
        return [client_list], 1
