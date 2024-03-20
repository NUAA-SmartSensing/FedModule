import copy


class GradientMixin:
    def __init__(self):
        self.global_model = None

    def _save_global_model(self, global_model: dict):
        self.global_model = copy.deepcopy(global_model)

    def _to_weights(self, grads: dict):
        new_model = {}
        for k, v in grads.items():
            new_model[k] = v + self.global_model[k]
        return new_model
