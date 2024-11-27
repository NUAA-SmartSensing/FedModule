from abc import abstractmethod


class AbstractPartitioner:
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def generate_data(self, iid_config, labels, client_num, dataset):
        pass
