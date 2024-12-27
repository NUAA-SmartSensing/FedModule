from typing import Dict, Any, Callable, Union, Optional

from utils import ModuleFindTool


class LossFactory:
    @staticmethod
    def create_loss(config: Union[str, Dict[str, Any]], belong_object: Optional[Any] = None, *args, **kwargs) -> Callable:
        # Check if the config is a string, indicating a direct class path
        if isinstance(config, str):
            loss_class = ModuleFindTool.find_class_by_path(config)
            return loss_class

        # Extract loss path and parameters from the config dictionary
        loss_path = config.get('path')
        loss_params = config.get('params', {})

        # Find the loss class using the provided path
        loss_class = ModuleFindTool.find_class_by_path(loss_path)
        from loss.AbstractLoss import AbstractLoss
        if issubclass(loss_class, AbstractLoss):
            return loss_class(belong_object, loss_params)
        else:
            return loss_class(*args, **loss_params, **kwargs)
