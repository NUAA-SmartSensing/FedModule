from utils import ModuleFindTool


class LossFactory:
    def __init__(self, config, belong_object=None):
        self.config = config
        self.belong_object = belong_object

    def create_loss(self, *args, **kwargs):
        if isinstance(self.config, str):
            loss_func = ModuleFindTool.find_class_by_path(self.config)
        else:
            if self.config['path'] == 'loss.FedLC.FedLC':
                return ModuleFindTool.find_class_by_path(self.config['path'])(self.config['params'], self.belong_object)
            if 'type' in self.config:
                if self.config['type'] == 'func':
                    loss_func = ModuleFindTool.find_class_by_path(self.config['path'])(**self.config['params'])
                else:
                    loss_class = ModuleFindTool.find_class_by_path(self.config['path'])
                    loss_func = loss_class(self.config['params'], *args, **kwargs)
            else:
                loss_class = ModuleFindTool.find_class_by_path(self.config['path'])
                loss_func = loss_class(self.config['params'], *args, **kwargs)
        return loss_func
