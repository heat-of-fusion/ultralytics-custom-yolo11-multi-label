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

import json
from collections import defaultdict
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import yaml
import numpy as np
import torch
from PIL import Image
from copy import deepcopy
from itertools import product
from torch.utils.data import ConcatDataset

from utils import verify_image_multi_label

from parameters import get_args_kwargs

from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM, colorstr
from ultralytics.utils.ops import resample_segments, segments2boxes
from ultralytics.utils.torch_utils import TORCHVISION_0_18

from utils import VectorIndexer

from augment import (
    Compose,
    Format,
    Instances,
    LetterBox,
    RandomLoadText,
    classify_augmentations,
    classify_transforms,
    v8_transforms,
)
from dataset_base import BaseDataset
from ultralytics.data.converter import merge_multi_segment
from ultralytics.data.utils import (
    HELP_URL,
    LOGGER,
    check_file_speeds,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
    save_dataset_cache_file,
    verify_image,
    verify_image_label,
)

# Ultralytics dataset *.cache version, >= 1.0.0 for YOLOv8
DATASET_CACHE_VERSION = "1.0.3"

##### MODIFIED FROM "ultralytics.data.dataset.YOLODataset"
class YOLODatasetMultiLabel(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    This class supports loading data for object detection, segmentation, pose estimation, and oriented bounding box
    (OBB) tasks using the YOLO format.

    Attributes:
        use_segments (bool): Indicates if segmentation masks should be used.
        use_keypoints (bool): Indicates if keypoints should be used for pose estimation.
        use_obb (bool): Indicates if oriented bounding boxes should be used.
        data (dict): Dataset configuration dictionary.

    Methods:
        cache_labels: Cache dataset labels, check images and read shapes.
        get_labels: Returns dictionary of labels for YOLO training.
        build_transforms: Builds and appends transforms to the list.
        close_mosaic: Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations.
        update_labels_info: Updates label format for different tasks.
        collate_fn: Collates data samples into batches.

    Examples:
        >>> dataset = YOLODataset(img_path="path/to/images", data={"names": {0: "person"}}, task="detect")
        >>> dataset.get_labels()
    """

    def __init__(self, *args, data=None, task="detect", **kwargs):
        """
        Initialize the YOLODataset.

        Args:
            data (dict, optional): Dataset configuration dictionary.
            task (str): Task type, one of 'detect', 'segment', 'pose', or 'obb'.
            *args (Any): Additional positional arguments for the parent class.
            **kwargs (Any): Additional keyword arguments for the parent class.
        """

        ##### BEGIN: MODIFIED
        self._kwargs = kwargs
        ##### END: MODIFIED

        self.use_segments = task == "segment"
        self.use_keypoints = task == "pose"
        self.use_obb = task == "obb"
        self.data = data

        ##### BEGIN: MODIFIED
        dims = ([self.data['ncs'][self.data['main_idx']]] +
                self.data['ncs'][:self.data['main_idx']] +
                self.data['ncs'][self.data['main_idx'] + 1:])
        self.vector_indexer = VectorIndexer(dims, self.data)
        ##### END: MODIFIED

        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        super().__init__(*args, **kwargs)

    ##### BEGIN: MODIFIED
    def get_image_and_label_with_processing(self, index):
        _ret = self.get_image_and_label(index)

        del _ret['cls'], _ret['cls_sub']
        _ret['cls'] = _ret['cls_for_ts']
        del _ret['cls_for_ts']

        return _ret
    ##### END: MODIFIED

    ##### BEGIN: MODIFIED
    def __getitem__(self, index):
        """Return transformed label information for given index."""

        _ret = self.get_image_and_label(index)

        del _ret['cls'], _ret['cls_sub']
        _ret['cls'] = _ret['cls_for_ts']
        del _ret['cls_for_ts']

        _ret = self.transforms(_ret)

        indices = self.vector_indexer.index_to_vector(_ret['cls'])

        _ret['cls'], _ret['cls_sub'] = indices[:, :1], indices[:, 1:]

        return _ret
    ##### END: MODIFIED

    def cache_labels(self, path=Path("./labels.cache")):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): Path where to save the cache file.

        Returns:
            (dict): Dictionary containing cached labels and related information.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in {2, 3}):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )

        ##### BEGIN: MODIFIED
        shift, main_shifter, sub_shifter = int(), int(), list()
        for i in range(self.data['nl']):
            if i != self.data['main_idx']:
                sub_shifter.append(shift)
            else:
                main_shifter = shift
            shift += self.data['ncs'][i]
        sub_shifter = np.array(sub_shifter).reshape(1, -1)
        print(f'>>>>>MAIN SHIFT>>>>>> {main_shifter}')
        print(f'>>>>>SUB SHIFT>>>>>> {sub_shifter}')
        ##### END: MODIFIED

        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(

                ##### BEGIN: MODIFIED
                func=verify_image_multi_label,
                ##### END: MODIFIED

                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                    repeat(self.single_cls),

                    ##### BEGIN: MODIFIED
                    repeat(self.data),
                    ##### END: MODIFIED
                ),
            )
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f

                if im_file:
                    ##### BEGIN: MODIFIED
                    main_idx = self.data['main_idx']
                    main_label = lb[:, main_idx:main_idx + 1] + main_shifter
                    sub_label = lb[:, self.data['sub_idxs']] + sub_shifter

                    ts_label = self.vector_indexer.vector_to_index(np.concatenate([main_label, sub_label], axis = -1))
                    ##### END: MODIFIED

                    x["labels"].append(
                        {
                            "im_file": im_file,
                            "shape": shape,
                            "cls": main_label,  # n, 1

                            ##### BEGIN: MODIFIED
                            "cls_sub": sub_label, # n, nl - 1
                            "cls_for_ts": ts_label,
                            ##### END: MODIFIED

                            "bboxes": lb[:, self.data['nl']:],  # n, 4
                            "segments": segments,
                            "keypoints": keypoint,
                            "normalized": True,
                            "bbox_format": "xywh",
                        }
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x, DATASET_CACHE_VERSION)
        return x

    def get_labels(self):
        """
        Returns dictionary of labels for YOLO training.

        This method loads labels from disk or cache, verifies their integrity, and prepares them for training.

        Returns:
            (List[dict]): List of label dictionaries, each containing information about an image and its annotations.
        """
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}")
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"WARNING ⚠️ No labels found in {cache_path}, training may not work correctly. {HELP_URL}")

        return labels

    ##### BEGIN: MODIFIED
    def recursive_shape(self, val, idx=0):
        if isinstance(val, list) or isinstance(val, tuple):
            for _val in val:
                self.recursive_shape(_val, idx + 1)
        elif isinstance(val, dict):
            for _val in val.keys():
                print('\t' * idx + _val)
                self.recursive_shape(val[_val], idx + 1)
        elif isinstance(val, str) or isinstance(val, int):
            print('\t' * idx + f'Type: {type(val)} | {val}')
        else:
            print(f'\t' * idx + f'Type: {type(val)} | {val.shape}')
            return
    ##### END: MODIFIED

    def build_transforms(self, hyp=None):
        """
        Builds and appends transforms to the list.

        Args:
            hyp (dict, optional): Hyperparameters for transforms.

        Returns:
            (Compose): Composed transforms.
        """
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms

    def close_mosaic(self, hyp):
        """
        Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations.

        Args:
            hyp (dict): Hyperparameters for transforms.
        """
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """
        Custom your label format here.

        Args:
            label (dict): Label dictionary containing bboxes, segments, keypoints, etc.

        Returns:
            (dict): Updated label dictionary with instances.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # make sure segments interpolate correctly if original length is greater than segment_resamples
            max_len = max(len(s) for s in segments)
            segment_resamples = (max_len + 1) if segment_resamples < max_len else segment_resamples
            # list[np.array(segment_resamples, 2)] * num_samples
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        """
        Collates data samples into batches.

        Args:
            batch (List[dict]): List of dictionaries containing sample data.

        Returns:
            (dict): Collated batch with stacked tensors.
        """
        new_batch = {}
        batch = [dict(sorted(b.items())) for b in batch]  # make sure the keys are in the same order
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k in {"img", "text_feats"}:
                value = torch.stack(value, 0)
            elif k == "visuals":
                value = torch.nn.utils.rnn.pad_sequence(value, batch_first=True)

            ##### BEGIN: MODIFIED
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb", 'cls_sub'}:
                value = torch.cat(value, 0)
            ##### END: MODIFIED

            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch

def build_yolo_dataset(data, split):
    '''
    Parameters:
        data: dataset configuration file dictionary from YAML format
        split: 'train', 'test' or 'valid'
    '''
    args, kwargs = get_args_kwargs(split)

    kwargs['augment'] = False if split in ['test', 'valid'] else True
    kwargs['hyp'].augment = False if split in ['test', 'valid'] else True

    dataset = YOLODatasetMultiLabel(*args, data = data, task = 'pose', **kwargs)

    return dataset

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    train_args, train_kwargs = get_args_kwargs('train')
    valid_args, valid_kwargs = get_args_kwargs('valid')

    data = yaml.safe_load(open(f'/home/pjh/VIDA/MMPatient/MMPatient/data_mmpatient.yaml', 'rb'))
    dataset = YOLODatasetMultiLabel(*train_args, data = data, task = 'pose', **train_kwargs)
