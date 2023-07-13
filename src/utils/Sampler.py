from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler


class DistributedSampler(Sampler[int]):
    def __init__(self, data_source: Sized, index_list: list, replacement: bool = False) -> None:
        self.generator = None
        self.replacement = replacement
        self.data_source = data_source
        self.index_list = index_list
        self.epoch = 0
        self._num_samples = len(index_list)

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.index_list)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.index_list)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from self.index_list[torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()]
            yield from self.index_list[torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()]
        else:
            for _ in range(self.num_samples // n):
                yield from self.index_list[torch.randperm(n, generator=generator).tolist()]
            yield from self.index_list[torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]]

    def __len__(self) -> int:
        return self.num_samples
