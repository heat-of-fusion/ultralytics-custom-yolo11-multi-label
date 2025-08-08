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
    # Get available device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Load parameters for training
    _, hyp = get_args_kwargs('train')

    # Load dataset configuration file
    data = yaml.safe_load(open(f'./data_mmpatient_full.yaml', 'rb'))

    # Get dataloaders for train, valid
    train_loader = get_dataloader(data, split = 'train')
    valid_loader = get_dataloader(data, split = 'valid')

    # Visualize batch
    batch = next(iter(train_loader))
    visualize_batch(batch, f'./history/batch0.png')

    # Get YOLO11-POSE model
    model = MMPatientYoloModel(f'./yolo11-pose.yaml', hyp, data).to(device)

    # Set loss function
    criterion = v8PoseLoss(model, data = data)

    # Build optimizer
    optimizer = build_optimizer(
        model = model.model,
        data = data,
        name = hyp['hyp'].optimizer,
        lr = hyp['hyp'].lr0,
        momentum = hyp['hyp'].momentum,
        decay = hyp['hyp'].weight_decay,
    )

    # Initialize history
    history = init_history(
        'box_loss',
        'pose_loss',
        'kobj_loss',
        'cls_loss',
        'dfl_loss',
        'total_loss'
    )

    # Initialize lowest validation loss
    lowest_valid_loss = np.inf

    # Generate directory for checkpoint
    if not os.path.isdir(f'history/model/'):
        os.mkdir(f'history/model/')

    # Start training loop
    for epoch in range(hyp['hyp'].epochs):

        # Set model to train mode
        model.train()

        # Initizlize train loss for logging
        train_loss = torch.zeros(5)

        for idx, batch in enumerate(tqdm(train_loader, desc = f'Epoch {epoch} | Training...')):

            # Preprocess batch
            batch = preprocess_batch(batch, hyp, device)

            # Model prediction
            pred = model(batch['img'])

            # Calculate loss
            loss, loss_items = criterion(pred, batch)
            loss = loss.sum()

            # Update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate train loss for logging
            train_loss += loss_items

        # Calculate average train loss
        train_loss = train_loss / len(train_loader)

        # Update train history
        history['train']['box_loss'].append(train_loss[0])
        history['train']['pose_loss'].append(train_loss[1])
        history['train']['kobj_loss'].append(train_loss[2])
        history['train']['cls_loss'].append(train_loss[3])
        history['train']['dfl_loss'].append(train_loss[4])
        history['train']['total_loss'].append(train_loss.sum())

        # Set model to evaluation mode
        model.eval()

        # Initialize valid loss for logging
        valid_loss = torch.zeros(5)

        with torch.no_grad():
            for idx, batch in enumerate(tqdm(valid_loader, desc = f'Epoch {epoch} | Validating...')):

                # Preprocess batch
                batch = preprocess_batch(batch, hyp, device)

                # Model prediction
                pred = model(batch['img'])

                # Calculate loss
                _, loss_items = criterion(pred, batch)

                # Accumulate valid loss for logging
                valid_loss += loss_items

        # Calculate average valid loss
        valid_loss = valid_loss / len(valid_loader)

        # Update valid history
        history['valid']['box_loss'].append(valid_loss[0])
        history['valid']['pose_loss'].append(valid_loss[1])
        history['valid']['kobj_loss'].append(valid_loss[2])
        history['valid']['cls_loss'].append(valid_loss[3])
        history['valid']['dfl_loss'].append(valid_loss[4])
        history['valid']['total_loss'].append(valid_loss.sum())

        # Log train, valid loss of current epoch
        print('-=' * 20 + f'\nEpoch {epoch}')
        print(f'Train | ' + ', '.join([f'{key}: {val:.5f}' for key, val in zip(loss_keys, train_loss[:5])]) + f', Total: {train_loss.sum():.5f}')
        print(f'Valid | ' + ', '.join([f'{key}: {val:.5f}' for key, val in zip(loss_keys, valid_loss[:5])]) + f', Total: {valid_loss.sum():.5f}')

        # Model checkpoint
        if lowest_valid_loss > history['valid']['total_loss'][-1]:
            print(f'!!! Lowest Validation Loss ReNewed! {lowest_valid_loss:.5f} -> {history["valid"]["total_loss"][-1]:.5f}')

            lowest_valid_loss = history['valid']['total_loss'][-1]
            torch.save(model.state_dict(), f'./history/model/Epoch_{epoch}_best_model_{model.cfg["scale"]}.pth')

        # Save log
        pkl.dump(history, open(f'./history/history.pkl', 'wb'))
        visualize_history(history, f'./history/')