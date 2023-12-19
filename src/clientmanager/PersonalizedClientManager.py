from clientmanager.BaseClientManager import BaseClientManager


class PersonalizedClientManager(BaseClientManager):
    def __init__(self, stop_event, config):
        BaseClientManager.__init__(self, stop_event, config)
        self.test_dataset = self.global_var["dataset"].get_test_dataset()

    def init_clients(self):
        return
