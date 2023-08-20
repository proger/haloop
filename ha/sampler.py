import torch
from torch.utils.data import Sampler
from typing import Iterator
import sys


class DurationBatchSampler(Sampler[list[int]]):
    def __init__(self, data_source, max_duration=240):
        self.data_source = data_source
        self.indices = torch.arange(len(data_source))
        self.max_duration = max_duration

    def __iter__(self) -> Iterator[list[int]]:
        batch: list[int] = []
        max_duration = 0
        for i in self.indices.tolist():
            sample_duration = self.data_source.duration(i)
            # use max duration of the batch to account for padding
            new_max_duration = max(max_duration, sample_duration)
            if (len(batch) + 1) * new_max_duration > self.max_duration:
                #print('batch', len(batch), len(batch)*max_duration, file=sys.stderr)
                yield batch
                batch = [i]
                max_duration = sample_duration
            else:
                batch.append(i)
                max_duration = new_max_duration
        if batch:
            yield batch
