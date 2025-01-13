import importlib
import os
import tempfile
import urllib.request
from typing import Dict, Any, Callable

import torch
from torch import nn


def find_class_by_path(path: str):
    path_list = path.split(".")
    for i in reversed(list(range(len(path_list)))):
        entry = '.'.join(path_list[:i+1])
        try:
            module = importlib.import_module(entry)
        except ModuleNotFoundError:
            continue
        attr_list = path_list[i:] if i > 0 else path_list[i+1:]
        try:
            for package in attr_list:
                module = getattr(module, package)
            return module
        except AttributeError:
            pass
    raise Exception(f"Module {path} not found.")


def generate_object_by_path(path: str, params: dict, else_params=None):
    target_class = find_class_by_path(path)
    if else_params is not None:
        params = params.update(else_params)
    target_object = target_class(**params)
    return target_object


def load_model_from_config(config: Dict[str, Any], src_obj=None) -> nn.Module:
    """
    从配置字典中加载模型。

    参数:
    - config (dict): 包含模型配置信息的字典

    返回:
    - model (nn.Module): 加载的模型
    """
    path = config.get("path", "")
    params = config.get("params", {})
    pretrained = config.get("pretrained", False)
    pretrained_path = config.get("pretrained_path", "")
    custom_create_fn = config.get("custom_create_fn", "")

    if custom_create_fn:
        # 使用用户自定义的创建函数
        create_fn: Callable = find_class_by_path(custom_create_fn)
        model = create_fn(src_obj, **params)
    else:
        if not path:
            raise ValueError("模型路径 'model.path' 不能为空。")

        # 动态导入模型类
        model_class = find_class_by_path(path)

        # 创建模型实例
        for k, v in params.items():
            if isinstance(v, str):
                params[k] = eval(v)
        model = model_class(**params)

        # 加载预训练权重
        if pretrained or pretrained_path:
            if pretrained:
                # 尝试使用模型类的预训练参数（适用于 PyTorch 内置模型）
                if hasattr(model_class, 'from_pretrained'):
                    # 对于支持 from_pretrained 的模型（如 HuggingFace 模型）
                    model = model_class.from_pretrained(path, **params)
                else:
                    # 对于 PyTorch 提供的模型
                    model = model_class(pretrained=True, **params)

            if pretrained_path:
                state_dict = load_pretrained_weights(pretrained_path)
                model.load_state_dict(state_dict)

    return model


def load_pretrained_weights(pretrained_path: str) -> Dict[str, Any]:
    """
    加载预训练权重，无论是本地路径还是URL。

    参数:
    - pretrained_path (str): 预训练权重的本地路径或URL

    返回:
    - state_dict (dict): 模型的 state_dict
    """
    if os.path.isfile(pretrained_path):
        # 本地文件
        state_dict = torch.load(pretrained_path, map_location='cpu')
    else:
        # 假设是URL，下载权重
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            print(f"下载预训练权重从 {pretrained_path} 到 {tmp_file.name}")
            urllib.request.urlretrieve(pretrained_path, tmp_file.name)
            state_dict = torch.load(tmp_file.name, map_location='cpu')
            os.remove(tmp_file.name)
    return state_dict
