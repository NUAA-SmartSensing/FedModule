class NoSchedule:
    def __init__(self):
        self.init = False

    def schedule(self, client_list, params):
        if self.init:
            return []
        else:
            self.init = True
            return client_list
