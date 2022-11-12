import math
import numpy as np
import torch
import torch.utils.data
from typing import Sized, Iterator
from torch.utils.data import Dataset, Sampler


class FirstLastSampler(Sampler):
    """
    A sampler that returns elements in a first-last order.
    """

    def __init__(self, data_source: Sized):
        """
        :param data_source: Source of data, can be anything that has a len(),
        since we only care about its number of elements.
        """
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        forward = 0
        backward = len(self.data_source)-1
        forward_step = True
        while(forward <= backward):
            if forward_step:
                idx = forward
                forward += 1
            else:
                idx = backward
                backward -= 1
            forward_step = not forward_step
            yield idx

    def __len__(self):
        return len(self.data_source)


class IndicesSampler(Sampler):

    def __init__(self, indices: Sized):
        super().__init__(indices)
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        idx = -1
        N = len(self.indices)
        while idx < N-1:
            idx += 1
            yield self.indices[idx]

    def __len__(self):
        return len(self.indices)


def create_train_validation_loaders(
    dataset: Dataset, validation_ratio, batch_size=100, num_workers=2
):
    """
    Splits a dataset into a train and validation set, returning a
    DataLoader for each.
    :param dataset: The dataset to split.
    :param validation_ratio: Ratio (in range 0,1) of the validation set size to
        total dataset size.
    :param batch_size: Batch size the loaders will return from each set.
    :param num_workers: Number of workers to pass to dataloader init.
    :return: A tuple of train and validation DataLoader instances.
    """
    if not (0.0 < validation_ratio < 1.0):
        raise ValueError(validation_ratio)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_ratio * dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = IndicesSampler(train_indices)
    valid_sampler = IndicesSampler(val_indices)

    dl_train = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler, num_workers=num_workers)
    dl_valid = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=valid_sampler, num_workers=num_workers)

    return dl_train, dl_valid
