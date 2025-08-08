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

import yaml
import torch
import numpy as np
from math import prod

import torch.nn as nn

from bbox_toolkit import *
from utils import recursive_shape

from ultralytics.nn.tasks import parse_model
from ultralytics.nn.modules.head import Detect, Pose

build_mmpatient_model = lambda cfg: MMPatientYoloModel(cfg)

class MMPatientYoloModel(nn.Module):
    def __init__(self, cfg, args, data, ch = 3):
        '''
        parameters:
        - cfg: Model configuration file
        - args: Parameters to compose model and predictions
        - ch: Number of input channels, e.g. 3(RGB)
        '''
        super(MMPatientYoloModel, self).__init__()

        # Save arguments to compose model and predictions
        self.cfg = yaml.safe_load(open(cfg, 'rb'))
        self.args = args
        self.data = data
        self.ch = ch

        # Parse configuration file and compose yolo model
        self.model = parse_model(self.cfg, ch = 3)[0]

        # Get additional parameters to compose predictions
        self.nc = self.cfg['nc']
        self.reg_max = self.model[-1].reg_max
        self.no = self.nc + self.reg_max * 4

        # Buffer to save prediction of each layer
        self.y = list()

        # Build stride tensor for each output featuremap of the yolo model
        self.build_stride()

        # Set names of the each classes indices
        self.model.names = {i: f'class{i}' for i in range(self.cfg['nc'])}

        # Projection tensor used when processing DFL
        self.proj = torch.arange(self.reg_max, dtype=torch.float32)

        # Main class index range
        self.Mm = sum(self.data['ncs'][:self.data['main_idx']])
        self.MM = self.Mm + self.data['ncs'][self.data['main_idx']]

        return

    def build_stride(self):
        '''
        Build Stride tensor for each output featuremap of the yolo model
        '''
        m = self.model[-1]
        if isinstance(m, Detect):
            s = 256

            def _forward(x):
                return self.forward(x)[0] if isinstance(m, Pose) else self.forward(x)

            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, self.ch, s, s))])
            self.stride = m.stride
            m.bias_init()
        else:
            self.stride = torch.Tensor([32])

        return

    def forward(self, x, inference = False):
        '''
        parameters:
        - x: torch.Tensor, Shape: [Batch, IN_CH, H, W]
        - inference: Boolean, Whether to apply post-process or not
        '''
        self.model[-1].training = False if inference else True

        # Initiate output buffer
        self.y = list()

        for idx, m in enumerate(self.model):
            # If the input of this layer is not from the layer right before, collect vectors from previous layers and compose input vector
            if m.f != -1:
                x = self.y[m.f] if isinstance(m.f, int) else [x if j == -1 else self.y[j] for j in m.f]

            # Process current layer
            x = m(x)

            # Save output
            self.y.append(x)

        return x

    def inference(self, x, post_process = True):
        '''
        parameters:
        - x: torch.Tensor, Shape: [Batch, IN_CH, H, W]
        - post_process: Boolean, Whether to apply post-process or not
        '''
        # Process forward with inference mode
        preds = self.forward(x, inference = True)[0]

        # Post Processing
        if post_process:
            # Detach predictions and change axes
            preds = preds.cpu().detach().permute(0, 2, 1)

            # Split prediction into bboxes, scores, keypoints
            _pred_bboxes, _pred_scores, _pred_kpts = preds.split([4, self.nc, prod(self.cfg['kpt_shape'])], dim = -1)

            # Filter background bboxes
            _pred_bboxes, _pred_scores, _pred_kpts = filter_anchors(_pred_bboxes, _pred_scores, _pred_kpts, self.data, threshold = 0.5)

            # Delete overlapping anchor indices through Non-Maximum Suppression
            nms_indices = batched_nms(_pred_bboxes, _pred_scores, [self.Mm, self.MM])

            # Final prediction buffer
            pred_bboxes, pred_scores, pred_kpts = list(), list(), list()

            # Leave only positive anchors for each prediction
            for idx in range(len(_pred_bboxes)):
                pred_bboxes.append(_pred_bboxes[idx][nms_indices[idx]].numpy().astype(np.int32))
                pred_scores.append(_pred_scores[idx][nms_indices[idx]].numpy().astype(np.float32))
                pred_kpts.append(_pred_kpts[idx][nms_indices[idx]].numpy().astype(np.int32))

            return pred_bboxes, pred_scores, pred_kpts

        return preds


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dummy_input = torch.randn([64, 3, 640, 640]).to(device)
    model = MMPatientYoloModel(f'./yolo11-mmpatient.yaml', None).to(device)

    recursive_shape(model(dummy_input))