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
import cv2
import random
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from ultralytics.utils.ops import segments2boxes
from ultralytics.data.utils import IMG_FORMATS, FORMATS_HELP_MSG, exif_size

loss_keys = ['BBox', 'Pose', 'KObj', 'Cls', 'DFL', 'Total']

def recursive_shape(val, idx=0):
    if isinstance(val, list) or isinstance(val, tuple):
        for _val in val:
            recursive_shape(_val, idx + 1)
    elif isinstance(val, dict):
        for _val in val.keys():
            print('\t' * idx + _val)
            recursive_shape(val[_val], idx + 1)
    elif isinstance(val, str) or isinstance(val, int):
        print('\t' * idx + f'Type: {type(val)} | {val}')
    else:
        print(f'\t' * idx + f'Type: {type(val)} | {val.shape}')
        return

##### MODIFIED FROM "ultralytics.engine.trainer.BaseTrainer.build_optimizer"
def build_optimizer(model, data, name = 'auto', lr = 0.001, momentum = 0.9, decay = 1e-5, iterations = 111400):

    ##### BEGIN: MODIFIED
    g = list(), list(), list()
    ##### END: MODIFIED

    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)

    if name == 'auto':

        ##### BEGIN: MODIFIED
        nc = data['nc']
        ##### END: MODIFIED

        lr_fit = round(0.002 * 5 / (4 + nc), 6)

        name, lr, momentum = 'AdamW', lr_fit, 0.9

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse = False):
            fullname = f'{module_name}.{param_name}' if module_name else param_name
            if 'bias' in fullname: # Bias -> No Decay
                g[2].append(param)
            elif isinstance(module, bn) or 'logit_scale' in fullname: # BN -> No Decay
                g[1].append(param)
            else: # Weight -> With Decay
                g[0].append(param)

    ##### BEGIN: MODIFIED
    optimizers = {'Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', 'SGD', 'auto'}
    ##### END: MODIFIED

    if name in {'Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam'}:
        optimizer = getattr(optim, name, optim.Adam)(g[2], lr = lr, betas = (momentum, 0.999), weight_decay = 0.0)
    elif name == 'RMSProp':
        optimizer = optim.RMSProp(g[2], lr = lr, momentum = momentum)
    elif name == 'SGD':
        optimizer = optim.SGD(g[2], lr = lr, momentum = momentum, nestrov = True)
    else:

        ##### BEGIN: MODIFIED
        raise NotImplementedError(f'Optimizer {name} not found.')
        ##### END: MODIFIED

    optimizer.add_param_group({'params': g[0], 'weight_decay': decay})
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})

    return optimizer

def preprocess_batch(batch, hyp, device):
    batch['img'] = batch['img'].to(device).float() / 255.

    # Preprocessing Script

    return batch

def transplant_weights(src_model, trg_model):
    print(trg_model.state_dict().keys())
    print(f'=-==-----=-=-=---===-=-===')
    if isinstance(src_model, str):
        src_model = torch.load(src_model)['model']
    print(src_model.state_dict().keys())
    return trg_model.load_state_dict(src_model.state_dict(), strict = False)

def init_history(*args):
    history = dict()
    history['args'] = args

    for split in ['train', 'valid']:
        history[split] = dict()
        for arg in args:
            history[split][arg] = list()

    return history

##### MODIFIED FROM "ultralytics.data.utils.verify_image_label"
def verify_image_multi_label(args): # Args: ..., data
    """Verify one image-label pair."""
    im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim, single_cls, data = args
    # Number (missing, found, empty, corrupt), message, segments, keypoints
    nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, "", [], None
    try:
        # Verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        shape = (shape[1], shape[0])  # hw
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}. {FORMATS_HELP_MSG}"
        if im.format.lower() in {"jpg", "jpeg"}:
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, "JPEG", subsampling=0, quality=100)
                    msg = f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"

        # Verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file, encoding="utf-8") as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]

                ##### BEGIN: MODIFIED
                lb = np.array(lb, dtype=np.float32)
                ##### END: MODIFIED
            if nl := len(lb):

                ##### BEGIN: MODIFIED
                if keypoint:
                    assert lb.shape[1] == ((4 + data['nl']) + nkpt * ndim), f"labels require {((data['nl'] + 4) + nkpt * ndim)} columns each"
                    points = lb[:, (data['nl'] + 4):].reshape(-1, ndim)[:, :2]
                else:
                    assert lb.shape[1] == (data['nl'] + 4), f"labels require {(data['nl'] + 4)} columns, {lb.shape[1]} columns detected"
                    points = lb[:, 1:]
                ##### END: MODIFIED
                assert points.max() <= 1, f"non-normalized or out of bounds coordinates {points[points > 1]}"
                assert lb.min() >= 0, f"negative label values {lb[lb < 0]}"

                # All labels
                if single_cls:
                    lb[:, 0] = 0
                max_cls = lb[:, 0].max()  # max label count
                assert max_cls < num_cls, (
                    f"Label class {int(max_cls)} exceeds dataset class count {num_cls}. "
                    f"Possible class labels are 0-{num_cls - 1}"
                )
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f"{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1  # label empty

                ##### BEGIN: MODIFIED
                lb = np.zeros((0, ((data['nl'] + 4) + nkpt * ndim) if keypoint else 5), dtype=np.float32)
                ##### END: MODIFIED
        else:
            nm = 1  # label missing

            ##### BEGIN: MODIFIED
            lb = np.zeros((0, ((data['nl'] + 4) + nkpt * ndim) if keypoints else 5), dtype=np.float32)
            ##### END: MODIFIED
        if keypoint:

            ##### BEGIN: MODIFIED
            keypoints = lb[:, (data['nl'] + 4):].reshape(-1, nkpt, ndim)
            ##### END: MODIFIED

            if ndim == 2:
                kpt_mask = np.where((keypoints[..., 0] < 0) | (keypoints[..., 1] < 0), 0.0, 1.0).astype(np.float32)
                keypoints = np.concatenate([keypoints, kpt_mask[..., None]], axis=-1)  # (nl, nkpt, 3)

        ##### BEGIN: MODIFIED
        lb = lb[:, :(data['nl'] + 4)]
        ##### END: MODIFIED

        return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, None, nm, nf, ne, nc, msg]

class VectorIndexer():
    def __init__(self, dims, data):
        self.dims = dims
        self.data = data

        self.weights = [1. for _ in range(len(dims))]
        for idx in reversed(range(len(dims) - 1)):
            self.weights[idx] = self.weights[idx + 1] * self.dims[idx + 1]

        self.weights = np.array(self.weights).reshape(1, -1)
        self.dims = np.array(dims).reshape(1, -1)

        self.shifter = torch.tensor([self.data['start_idxs'][self.data['main_idx']]] +
                        self.data['start_idxs'][:self.data['main_idx']] +
                        self.data['start_idxs'][self.data['main_idx'] + 1:]).view(1, -1)

        print(f'Indexer Shifter: {self.shifter}')

        return

    def vector_to_index(self, x):
        if not isinstance(x, torch.Tensor):
            shifter = self.shifter.numpy()

        x -= shifter

        if len(x.shape) == 1:
            vector = x.reshape(1, -1)

        index = (x * self.weights).sum(axis = -1).astype(np.int_)

        return index

    def index_to_vector(self, index):
        return_torch = False

        if isinstance(index, torch.Tensor):
            return_torch = True
            index = index.numpy()
            shifter = self.shifter
        else:
            shifter = self.shifter.numpy()

        if len(index.shape) == 1:
            index = index.reshape(-1, 1)  # [N, 1]

        n = index.shape[0]
        d = self.dims.shape[-1]
        vector = np.zeros((n, d)).astype(np.int_)

        index = index.copy()

        for idx in range(d):
            w = self.weights[0, idx]
            vector[:, idx] = index[:, 0] // w
            index[:, 0] = index[:, 0] % w

        return torch.tensor(vector) + shifter if return_torch else vector + shifter

if __name__ == '__main__':
    print(init_history('box_loss',
        'pose_loss',
        'kobj_loss',
        'cls_loss',
        'dfl_loss',
        'total_loss'))