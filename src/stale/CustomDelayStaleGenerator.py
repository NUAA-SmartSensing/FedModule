import random
import numpy as np

from utils import ModuleFindTool


def uniform_delay(min_delay, max_delay):
    def generator():
        return random.uniform(min_delay, max_delay)
    return generator


def normal_delay(mean, std, min_delay, max_delay):
    def generator():
        d = np.random.normal(mean, std)
        d = max(min_delay, min(max_delay, d))
        return d
    return generator


def exponential_delay(lam, min_delay=0.0, max_delay=10.0):
    def generator():
        d = np.random.exponential(lam)
        d = max(min_delay, min(max_delay, d))
        return d
    return generator


def get_delay_generator_by_type(type_name, params):
    if type_name == "uniform":
        return uniform_delay(params["min_delay"], params["max_delay"])
    elif type_name == "normal":
        return normal_delay(params["mean"], params["std"], params["min_delay"], params["max_delay"])
    elif type_name == "exponential":
        return exponential_delay(params["lam"], params.get("min_delay", 0.0), params.get("max_delay", 10.0))
    elif type_name == "constant":
        return params["value"]
    else:
        try:
            return ModuleFindTool.find_class_by_path(params["path"])()(params["params"])
        except ModuleNotFoundError:
            raise ValueError(f"Unknown delay type: {type_name}")


class CustomDelayStaleGenerator:
    """
    支持为不同客户端分配不同延迟分布的生成器。
    params: {
        "client_num": int,
        "groups": [
            {"num": 3, "type": "uniform", "min_delay": 0, "max_delay": 1},
            {"num": 5, "type": "normal", "mean": 2.0, "std": 0.5, "min_delay": 0.1, "max_delay": 5.0},
            {"num": 5, "type": "normal", "mean": 5.0, "std": 1.0, "min_delay": 2.0, "max_delay": 8.0}
        ]
    }
    """

    def __init__(self, params):
        self.client_num = params["client_num"]
        self.groups = params["groups"]
        self.delay_generators = self._build_delay_generators()

    def _build_delay_generators(self):
        gens = []
        for group in self.groups:
            for _ in range(group["num"]):
                gen = get_delay_generator_by_type(group["type"], group)
                gens.append(gen)
        # 若数量不足，补齐
        while len(gens) < self.client_num:
            gens.append(0)
        return gens[:self.client_num]

    def generate_staleness_list(self):
        """
        返回每个客户端的延迟生成器列表
        """
        return self.delay_generators
