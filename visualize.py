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

import cv2
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

def visualize_batch(batch, savedir):
    plt.figure(figsize = (8, 8))
    for grid_y in range(4):
        for grid_x in range(4):
            ord = ((grid_y * 4) + (grid_x + 1))
            plt.subplot(4, 4, ord)
            plt.axis(f'off')

            img = (batch['img'][ord - 1].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img = np.ascontiguousarray(img)
            shape = np.array(img.shape[:2][::-1])

            bboxes = batch['bboxes'].numpy()[np.where(batch['batch_idx'].numpy() == (ord - 1))]
            for bbox in bboxes:
                x, y = (bbox[:2] * shape).astype(np.int32)
                w, h = (bbox[2:] * shape).astype(np.int32)

                cv2.rectangle(img, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 0, 255), 10)

            plt.imshow(img)

            kps = batch['keypoints'].numpy()[np.where(batch['batch_idx'].numpy() == (ord - 1))]
            for kp in kps:
                kp = kp[:, :2] * shape[None]
                plt.scatter(kp[:, 0], kp[:, 1], s = 1)

            plt.title(f'Img[{ord}]')
    plt.savefig(savedir)
    plt.close()

    return

def visualize_history(history, save_dir = f'./history/'):
    if isinstance(history, str): # Case 0: Visualize from file
        history = pkl.load(open(history, 'rb'))

    args = history['args']

    plt.figure(figsize = (5 * len(args), 4))
    for idx, arg in enumerate(args):
        plt.subplot(1, len(args), idx + 1)
        plt.plot(history['train'][arg], label = 'Train')
        plt.plot(history['valid'][arg], label = 'Valid')
        plt.xlabel(f'Epoch')
        plt.ylabel(f'Loss')
        plt.title(arg)
        plt.legend()

    plt.savefig(save_dir + f'history.png')
    plt.close()

    return

if __name__ == '__main__':
    visualize_history(f'./history/history.pkl')