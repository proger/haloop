import torch
from torch.utils.data import Sampler
from typing import Iterator


class DurationBatchSampler(Sampler[list[int]]):
    def __init__(self, data_source, max_duration=240):
        self.data_source = data_source
        self.indices = torch.arange(len(data_source))
        self.max_duration = max_duration

    def __iter__(self) -> Iterator[list[int]]:
        batch: list[int] = []
        duration, max_duration = 0, 0
        for i in self.indices.tolist():
            sample_duration = self.data_source.duration(i)
            if (duration + sample_duration) > self.max_duration:
                #print('batch', len(batch), duration)
                yield batch
                batch = []
                duration, max_duration = 0, 0
            batch.append(i)
            # use max duration of the batch to account for padding
            max_duration = max(max_duration, sample_duration)
            duration = len(batch) * max_duration
