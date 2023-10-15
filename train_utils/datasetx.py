

from torch.utils.data import Dataset, DataLoader

import torch

# collapse-hide
import random
from torch.utils.data.sampler import Sampler

from torch.utils.data import DataLoader

# 构造数据
xs = list(range(11))
ys = list(range(10, 21))
print('xs values: ', xs)
print('ys values: ', ys)


# 定义Dataset子类
class MyDataset:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __getitem__(self, i):
        # return self.xs[i[0]], self.ys[i[0]]

        return self.xs[i], self.ys[i]

    def __len__(self):
        return len(self.xs)
dataset = MyDataset(xs, ys)
class IndependentHalvesSampler(Sampler):
    def __init__(self, dataset):
        halfway_point = int(len(dataset) / 2)
        self.first_half_indices = list(range(halfway_point))
        self.second_half_indices = list(range(halfway_point, len(dataset)))

    def __iter__(self):
        random.shuffle(self.first_half_indices)
        random.shuffle(self.second_half_indices)
        xs=iter(self.first_half_indices + self.second_half_indices)
        # xs = iter([(1,2),]*11)
        return xs

    def __len__(self):
        return len(self.first_half_indices) + len(self.second_half_indices)


class BatchSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.
    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """
# 批次采样
    def __init__(self, sampler, batch_size, drop_last=False):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        # if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
        #         batch_size <= 0:
        #     raise ValueError("batch_size should be a positive integeral value, "
        #                      "but got batch_size={}".format(batch_size))
        # if not isinstance(drop_last, bool):
        #     raise ValueError("drop_last should be a boolean value, but got "
        #                      "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


our_sampler = IndependentHalvesSampler(dataset)
print('First half indices: ', our_sampler.first_half_indices)
print('Second half indices:', our_sampler.second_half_indices)

dl = DataLoader(dataset  #, sampler=our_sampler, batch_size=1
                ,batch_sampler=BatchSampler(our_sampler,batch_size=1))

for i, data in enumerate(dl):
    print(data)

for i, data in enumerate(dl):
    print(data)


train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,num_replicas=2,rank=1)
dl = DataLoader(dataset  , sampler=train_sampler, batch_size=1
               # ,batch_sampler=BatchSampler(our_sampler,batch_size=1)
                )
for i, data in enumerate(dl):
    print(data)

