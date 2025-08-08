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

import torch
from torchvision.ops import nms

def xywh_to_xyxy(bboxes):
    '''
    Convert [XYWH] format boxes into [XYXY(ltrb)] format.
    parameters:
    - bboxes: Torch.Tensor, Size: [Batch, N_Anchors, 4]
    '''
    x1 = bboxes[..., 0] - bboxes[..., 2] / 2
    y1 = bboxes[..., 1] - bboxes[..., 3] / 2
    x2 = bboxes[..., 0] + bboxes[..., 2] / 2
    y2 = bboxes[..., 1] + bboxes[..., 3] / 2

    return torch.stack([x1, y1, x2, y2], dim = -1)

def batched_nms(bboxes, scores, mMrange = None, iou_threshold = 0.5):
    '''
    Remove common bboxes with Non-Maximum Suppression and get
    positive anchor indices.
    parameters:
    - bboxes: torch.Tensor, Size: [Batch, N_Anchors, 4]
    - scores: torch.Tensor, Size: [Batch, N_Anchors, N_CLS]
    - iou_threshold: Int, IOU threshold for NMS
    '''
    # Get batch size
    B = len(bboxes)

    # Positive anchor indices for each image from the batch
    keep_indices_per_batch = list()

    for b in range(B):
        # Convert [XYWH] box to [XYXY(ltrb)] box
        boxes_i = xywh_to_xyxy(bboxes[b]).float()

        # Get the maximum score idx for each anchor
        if isinstance(mMrange, list):
            scores_i = scores[b][:, mMrange[0]:mMrange[1]].argmax(dim = 1).float()
        else:
            scores_i = scores[b].argmax(dim = 1).float()

        # Process Non-Maximum Suppression
        keep = nms(boxes_i, scores_i, iou_threshold)

        # Save anchor indices to the buffer
        keep_indices_per_batch.append(keep)

    return keep_indices_per_batch

def filter_anchors(bboxes, scores, kpts, data, threshold = 0.5):
    '''
    Filtering predictions with the classification scores.
    If the maximum score is lower than the threshold,
    this function determines the prediction to background.
    parameters:
    - bboxes: torch.Tensor, Size: [Batch, N_Anchors, 4]
    - scores: torch.Tensor, Size: [Batch, N_Anchos, N_CLS]
    - kpts: torch.Tensor, Size: [Batch, N_Anchors, 15 * 3]
    - threshold: Int, Minimum bound of the classification score.
    '''
    # Get batch size
    B, _, _ = scores.shape

    # # Get the maximum score among total classes # Method:V0
    # max_scores, _ = scores.max(dim = 2)

    # # Get the maximum scores from each sub-labels and averaging them  # Method:V1
    # max_scores = list()
    # for idx in range(len(data['ncs'])):
    #     _max_scores, _ = scores[..., data['start_idxs'][idx]: data['start_idxs'][idx] + data['ncs'][idx]].max(dim=2, keepdim = True)
    #     max_scores.append(_max_scores)
    #
    # max_scores = torch.cat(max_scores, dim=-1).mean(dim=-1)

    # Get the maximum score of certain sub-label # Method:V2
    # M_LBL = 2
    # M_IDX = data['start_idxs'][M_LBL]
    # max_scores, _ = scores[..., M_IDX : M_IDX + data['ncs'][M_LBL]].max(dim = 2)

    # Get the maximum score of the main label # Method:V3
    M_LBL = data['main_idx']
    M_IDX = data['start_idxs'][M_LBL]
    max_scores, _ = scores[..., M_IDX : M_IDX + data['ncs'][M_LBL]].max(dim = 2)

    # Buffer to save filtered predictions
    filtered_bboxes_list = list()
    filtered_scores_list = list()
    filtered_kpts_list = list()

    for b in range(B):
        # Filtering anchors whose maximum score exceeds threshold
        mask = max_scores[b] >= threshold

        filtered_bboxes = bboxes[b][mask]
        filtered_scores = scores[b][mask]
        filtered_kpts = kpts[b][mask]

        # Save object anchors to each buffer
        filtered_bboxes_list.append(filtered_bboxes)
        filtered_scores_list.append(filtered_scores)
        filtered_kpts_list.append(filtered_kpts)

    return filtered_bboxes_list, filtered_scores_list, filtered_kpts_list