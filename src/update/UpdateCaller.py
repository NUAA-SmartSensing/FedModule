class UpdateCaller:
    def __init__(self, updater):
        self.updater = updater

    def update_server_weights(self, epoch, update_list):
        return self.updater.update_method.update_server_weights(epoch, update_list)
