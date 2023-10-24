import torch


def to_cpu(data):
    if isinstance(data, dict):
        return {k: to_cpu(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_cpu(v) for v in data]
    elif isinstance(data, torch.Tensor):
        return data.cpu().detach()
    else:
        return data


class UpdateCaller:
    def __init__(self, updater):
        self.updater = updater

    def update_server_weights(self, epoch, update_list, *args, **kwargs):
        # 确保返参进入CPU
        a, b = self.updater.update_method.update_server_weights(epoch, update_list)
        return to_cpu(a), to_cpu(b)
