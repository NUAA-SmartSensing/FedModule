import random
from typing import List, TypeVar, Optional
T = TypeVar('T')


def bernoulli_sampling(input_list: List[T], p: float, seed: Optional[int] = None) -> List[T]:
    """
    对输入列表进行伯努利采样

    参数:
        input_list: 输入列表，可以是任意类型的元素
        p: 采样概率，范围[0, 1]
        seed: 随机数种子，用于复现结果

    返回:
        采样后的列表

    示例:
        >>> lst = [1, 2, 3, 4, 5]
        >>> bernoulli_sampling(lst, 0.6, seed=42)
        [2, 3, 5]
    """
    if not 0 <= p <= 1:
        raise ValueError("采样概率 p 必须在 0 和 1 之间")

    if seed is not None:
        random.seed(seed)

    return [x for x in input_list if random.random() < p]
