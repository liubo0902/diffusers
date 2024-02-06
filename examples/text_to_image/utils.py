# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Sequence
from torch.utils.data import BatchSampler, Sampler, Dataset
from random import shuffle, choice
from copy import deepcopy

ASPECT_RATIO_512 = {
     '0.25': [256.0, 1024.0], '0.26': [256.0, 992.0], '0.27': [256.0, 960.0], '0.28': [256.0, 928.0],
     '0.32': [288.0, 896.0], '0.33': [288.0, 864.0], '0.35': [288.0, 832.0], '0.4': [320.0, 800.0],
     '0.42': [320.0, 768.0], '0.48': [352.0, 736.0], '0.5': [352.0, 704.0], '0.52': [352.0, 672.0],
     '0.57': [384.0, 672.0], '0.6': [384.0, 640.0], '0.68': [416.0, 608.0], '0.72': [416.0, 576.0],
     '0.78': [448.0, 576.0], '0.82': [448.0, 544.0], '0.88': [480.0, 544.0], '0.94': [480.0, 512.0],
     '1.0': [512.0, 512.0], '1.07': [512.0, 480.0], '1.13': [544.0, 480.0], '1.21': [544.0, 448.0],
     '1.29': [576.0, 448.0], '1.38': [576.0, 416.0], '1.46': [608.0, 416.0], '1.67': [640.0, 384.0],
     '1.75': [672.0, 384.0], '2.0': [704.0, 352.0], '2.09': [736.0, 352.0], '2.4': [768.0, 320.0],
     '2.5': [800.0, 320.0], '2.89': [832.0, 288.0], '3.0': [864.0, 288.0], '3.11': [896.0, 288.0],
     '3.62': [928.0, 256.0], '3.75': [960.0, 256.0], '3.88': [992.0, 256.0], '4.0': [1024.0, 256.0]
     }

def get_closest_ratio(height: float, width: float, ratios: dict = ASPECT_RATIO_512):
    aspect_ratio = height / width
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return ratios[closest_ratio], float(closest_ratio)


class AspectRatioBatchSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(self,
                 sampler: Sampler,
                 dataset: Dataset,
                 batch_size: int,
                 aspect_ratios: dict = ASPECT_RATIO_512,
                 drop_last: bool = False,
                 config=None,
                 valid_num=0,   # take as valid aspect-ratio when sample number >= valid_num
                 **kwargs) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.aspect_ratios = aspect_ratios
        self.drop_last = drop_last
        self.config = config
        # buckets for each aspect ratio
        self._aspect_ratio_buckets = {ratio: [] for ratio in aspect_ratios}
        self.current_available_bucket_keys = list(aspect_ratios.keys()) # [str(k) for k, v in aspect_ratios]
        self.ratios = self.dataset['ratio']

    def __iter__(self) -> Sequence[int]:
        for idx in self.sampler:
            # data_info = self.dataset[idx]['ratio']
            # height, width =  data_info['image'].size[1], data_info['image'].size[0]
            ratio = self.ratios[idx]
            # find the closest aspect ratio
            closest_ratio = min(self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio))
            if closest_ratio not in self.current_available_bucket_keys:
                continue
            bucket = self._aspect_ratio_buckets[closest_ratio]
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]

        # yield the rest data and reset the buckets
        for bucket in self._aspect_ratio_buckets.values():
            while len(bucket) > 0:
                if len(bucket) <= self.batch_size:
                    if not self.drop_last:
                        yield bucket[:]
                    bucket = []
                else:
                    yield bucket[:self.batch_size]
                    bucket = bucket[self.batch_size:]


class BalancedAspectRatioBatchSampler(AspectRatioBatchSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Assign samples to each bucket
        self.ratio_nums_gt = kwargs.get('ratio_nums', None)
        assert self.ratio_nums_gt
        self._aspect_ratio_buckets = {float(ratio): [] for ratio in self.aspect_ratios.keys()}
        self.original_buckets = {}
        self.current_available_bucket_keys =  [k for k, v in self.ratio_nums_gt.items() if v >= 3000]
        self.all_available_keys = deepcopy(self.current_available_bucket_keys)
        self.exhausted_bucket_keys = []
        self.total_batches = len(self.sampler) // self.batch_size
        self._aspect_ratio_count = {}
        for k in self.all_available_keys:
            self._aspect_ratio_count[float(k)] = 0
            self.original_buckets[float(k)] = []

    def __iter__(self) -> Sequence[int]:
        i = 0
        for idx in self.sampler:
            data_info = self.dataset.get_data_info(idx)
            height, width = data_info['height'], data_info['width']
            ratio = height / width
            closest_ratio = float(min(self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio)))
            if closest_ratio not in self.all_available_keys:
                continue
            if self._aspect_ratio_count[closest_ratio] < self.ratio_nums_gt[closest_ratio]:
                self._aspect_ratio_count[closest_ratio] += 1
                self._aspect_ratio_buckets[closest_ratio].append(idx)
                self.original_buckets[closest_ratio].append(idx)    # Save the original samples for each bucket
            if not self.current_available_bucket_keys:
                self.current_available_bucket_keys, self.exhausted_bucket_keys = self.exhausted_bucket_keys, []

            if closest_ratio not in self.current_available_bucket_keys:
                continue
            key = closest_ratio
            bucket = self._aspect_ratio_buckets[key]
            if len(bucket) == self.batch_size:
                yield bucket[:self.batch_size]
                del bucket[:self.batch_size]
                i += 1
                self.exhausted_bucket_keys.append(key)
                self.current_available_bucket_keys.remove(key)

        for _ in range(self.total_batches - i):
            key = choice(self.all_available_keys)
            bucket = self._aspect_ratio_buckets[key]
            if len(bucket) >= self.batch_size:
                yield bucket[:self.batch_size]
                del bucket[:self.batch_size]

                # If a bucket is exhausted
                if not bucket:
                    self._aspect_ratio_buckets[key] = deepcopy(self.original_buckets[key][:])
                    shuffle(self._aspect_ratio_buckets[key])
            else:
                self._aspect_ratio_buckets[key] = deepcopy(self.original_buckets[key][:])
                shuffle(self._aspect_ratio_buckets[key])
