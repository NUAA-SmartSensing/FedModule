import torch.cuda

from utils.Tools import to_cpu, to_dev


class UpdateCaller:
    def __init__(self, updater, update_method = None):
        self.updater = updater
        self.update_method = update_method if update_method is not None else updater.update_method

    def update_server_weights(self, epoch, update_list, *args, **kwargs):
        # 确保形参进入GPU
        if torch.cuda.is_available():
            update_list = to_dev(update_list, 'cuda')
        # 确保返参进入CPU
        result = self.update_method.update_server_weights(epoch, update_list)
        return to_cpu(result)

