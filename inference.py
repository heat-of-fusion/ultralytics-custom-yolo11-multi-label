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
import yaml
import numpy as np
import pickle as pkl
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch

from utils import *
from loss import v8PoseLoss
from model import MMPatientYoloModel
from dataloader import get_dataloader
from parameters import get_args_kwargs
from bbox_toolkit import batched_nms, filter_anchors

from visualize import visualize_batch

if __name__ == '__main__':
    # Get available device
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    # Load parameters for training
    _, hyp = get_args_kwargs('train')

    # Load dataaset configuration file
    data = yaml.safe_load(open(f'./data_mmpatient_full.yaml', 'rb'))

    # Get dataloader for test
    test_loader = get_dataloader(data, split = 'test')

    # Get YOLO11-POSE model
    model = MMPatientYoloModel(f'./yolo11-pose.yaml', hyp, data).to(device)

    # Load model parameters
    msg = model.load_state_dict(torch.load(f'./history/model/Epoch_282_best_model_s.pth'))

    print(f'Model: {msg}')

    # Generate directory for inference
    if not os.path.isdir(f'./history/inference/'):
        os.mkdir(f'./history/inference/')

    for batch in tqdm(test_loader, desc = f'Inferencing...'):

        # Preprocess batch
        batch = preprocess_batch(batch, hyp, device)

        # Model prediction
        pred_bboxes, pred_scores, pred_kpts = model.inference(batch['img'], post_process = True)

        # Get image
        img = np.ascontiguousarray((batch['img'].cpu().detach().permute(0, 2, 3, 1).numpy() * 255).clip(0.0, 255.0).astype(np.uint8))

        # Batch loop
        for b_idx in range(img.shape[0]):

            # Get image file name
            file_name = batch['im_file'][b_idx].split('/')[-1]

            # Anchor loop
            for a_idx in range(len(pred_bboxes[b_idx])):

                # Get BBox
                x, y, w, h = pred_bboxes[b_idx][a_idx]

                # Get keypoints
                kpt_coord = pred_kpts[b_idx][a_idx]

                # Get class names for each label
                cls_names, start = list(), int()
                for idx in range(data['nl']):
                    cls_names.append(data['names'][pred_scores[b_idx][a_idx][start:start + data['ncs'][idx]].argmax() + start])
                    start += data['ncs'][idx]

                # Join class names
                cls_name = ' \n '.join(cls_names)

                # XYWH -> XYXY
                bbox_pos = [x - w // 2, y - h // 2], [x + w // 2, y + h // 2]

                # Draw BBox
                cv2.rectangle(img[b_idx], bbox_pos[0], bbox_pos[1], (255, 0, 0), 1)

                # Draw keypoints
                kpt_coord = kpt_coord.reshape(15, 3)
                for k_idx in range(len(kpt_coord)):
                    cv2.circle(img[b_idx], center = kpt_coord[k_idx][:2], radius = 3, color = (255, 0, 0), thickness = -1)

                # Put class names into images
                cv2.putText(
                    img[b_idx],
                    cls_name,
                    [val - 10 for val in bbox_pos[0]],
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.3,
                    color=(255, 0, 0),
                    thickness=1
                )

            # Save image
            cv2.imwrite(f'./history/inference/{file_name}', cv2.cvtColor(img[b_idx], cv2.COLOR_RGB2BGR))