class RandomSchedule:
    def __init__(self):
        self.init = False

    def schedule(self, client_list, params):
        if self.init:
            self.init = True
            return []
        else:
            return client_list
