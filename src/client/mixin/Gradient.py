import copy


class GradientMixin:
    r"""
    Mixin class for the client to compute the gradient of the local model
    To support client upload the gradient instead of the model

    methods:

    - _save_global_model: save the global model
    - _to_gradient: compute the gradient of the local model, grad = local_model - global_model
    """
    def __init__(self):
        self.global_model = None

    def _save_global_model(self, global_model):
        r"""
        save the global model

        Parameters
        ----------
        global_model: dict
            the global model's state_dict
        """
        self.global_model = copy.deepcopy(global_model)

    def _to_gradient(self):
        r"""
        compute the gradient of the local model, grad = local_model - global_model

        [NOTE]
        please call this method after the local model has been updated, and local model is saved in self.model

        Returns
        -------
        dict
            the gradient of the local model
        """
        grad = {}
        weights = self.model.state_dict()
        for k, v in self.global_model.items():
            grad[k] = weights[k] - v
        return grad
