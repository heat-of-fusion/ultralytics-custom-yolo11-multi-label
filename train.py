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
from visualize import visualize_batch, visualize_history

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    _, hyp = get_args_kwargs('train')

    data = yaml.safe_load(open(f'./data_mmpatient_full.yaml', 'rb'))

    train_loader = get_dataloader(data, split = 'train')
    valid_loader = get_dataloader(data, split = 'valid')

    batch = next(iter(train_loader))
    print(f'<<<Pre-Test')
    print(type(batch['cls']))
    print(type(batch['cls_sub']))
    print(f'------')
    visualize_batch(batch, f'./history/batch0.png')

    # print(batch['cls'])
    # print(batch['cls_sub'])

    # exit()

    model = MMPatientYoloModel(f'./yolo11-pose.yaml', hyp, data).to(device)
    criterion = v8PoseLoss(model, data = data)

    optimizer = build_optimizer(
        model = model.model,
        data = data,
        name = hyp['hyp'].optimizer,
        lr = hyp['hyp'].lr0,
        momentum = hyp['hyp'].momentum,
        decay = hyp['hyp'].weight_decay,
    )

    history = init_history(
        'box_loss',
        'pose_loss',
        'kobj_loss',
        'cls_loss',
        'dfl_loss',
        'total_loss'
    )

    lowest_valid_loss = np.inf

    if not os.path.isdir(f'history/model/'):
        os.mkdir(f'history/model/')

    for epoch in range(hyp['hyp'].epochs):
        model.train()
        train_loss = torch.zeros(5)

        for idx, batch in enumerate(tqdm(train_loader, desc = f'Epoch {epoch} | Training...')):
            batch = preprocess_batch(batch, hyp, device)

            # print(batch['cls'])
            # print(batch['cls_sub'])

            pred = model(batch['img'])

            loss, loss_items = criterion(pred, batch)
            loss = loss.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss_items

        train_loss = train_loss / len(train_loader)

        history['train']['box_loss'].append(train_loss[0])
        history['train']['pose_loss'].append(train_loss[1])
        history['train']['kobj_loss'].append(train_loss[2])
        history['train']['cls_loss'].append(train_loss[3])
        history['train']['dfl_loss'].append(train_loss[4])
        history['train']['total_loss'].append(train_loss.sum())

        model.eval()
        valid_loss = torch.zeros(5)

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(valid_loader, desc = f'Epoch {epoch} | Validating...')):
                batch = preprocess_batch(batch, hyp, device)
                pred = model(batch['img'])

                _, loss_items = criterion(pred, batch)

                valid_loss += loss_items

        valid_loss = valid_loss / len(valid_loader)

        history['valid']['box_loss'].append(valid_loss[0])
        history['valid']['pose_loss'].append(valid_loss[1])
        history['valid']['kobj_loss'].append(valid_loss[2])
        history['valid']['cls_loss'].append(valid_loss[3])
        history['valid']['dfl_loss'].append(valid_loss[4])
        history['valid']['total_loss'].append(valid_loss.sum())

        print('-=' * 20 + f'\nEpoch {epoch}')
        print(f'Train | ' + ', '.join([f'{key}: {val:.5f}' for key, val in zip(loss_keys, train_loss[:5])]) + f', Total: {train_loss.sum():.5f}')
        print(f'Valid | ' + ', '.join([f'{key}: {val:.5f}' for key, val in zip(loss_keys, valid_loss[:5])]) + f', Total: {valid_loss.sum():.5f}')

        if lowest_valid_loss > history['valid']['total_loss'][-1]:
            print(f'!!! Lowest Validation Loss ReNewed! {lowest_valid_loss:.5f} -> {history["valid"]["total_loss"][-1]:.5f}')

            lowest_valid_loss = history['valid']['total_loss'][-1]
            torch.save(model.state_dict(), f'./history/model/Epoch_{epoch}_best_model_{model.cfg["scale"]}.pth')

        pkl.dump(history, open(f'./history/history.pkl', 'wb'))
        visualize_history(history, f'./history/')