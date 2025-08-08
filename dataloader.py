# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# Copyright (C) 2025 Ultralytics, Jehyeon Park
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import os
import torch
from torch.utils.data import dataloader, distributed

from dataset import build_yolo_dataset

from ultralytics.data.build import InfiniteDataLoader

def get_dataloader(data, batch_size = 16, split = 'train'):
    assert split in {'train', 'valid', 'test'}, f'Split must be "train" or "valid", not {mode}.'

    dataset = build_yolo_dataset(data, split)

    shuffle = split == 'train'

    if getattr(dataset, 'rect', False) and shuffle:
        print(f'"rect = True" is incompatible with DataLoader shuffle, setting shuffle = False')
        shuffle = False

    workers = dataset._kwargs['hyp'].workers

    return build_dataloader(dataset, batch_size, workers, shuffle)

def build_dataloader(dataset, batch_size, workers, shuffle = True):
    batch_size = min(batch_size, len(dataset))

    nd = torch.cuda.device_count()
    nw = min(os.cpu_count() // max(nd, 1), workers)

    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + int(os.getenv("RANK", -1)))

    return InfiniteDataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = nw,
        collate_fn = getattr(dataset, 'collate_fn', None),
    )