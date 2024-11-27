from abc import abstractmethod


class AbstractDataGen:
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def generate_data(self, iid_config, labels, client_num, dataset, train=True):
        pass
