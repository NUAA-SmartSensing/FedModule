from fedsemi.grouping.Grouping import Grouping


class SimpleGrouping(Grouping):
    def __init__(self, config):
        Grouping.__init__(self, config)

    def group(self, client_list, latency_list, config):
        return [client_list], 1
