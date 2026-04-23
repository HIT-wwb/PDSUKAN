import argparse
import os
import random
import time
from collections import OrderedDict
from glob import glob
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from albumentations import RandomRotate90, Resize
from albumentations.augmentations import geometric, transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
import losses
import PDSUKAN
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool

LOSS_NAMES = losses.__all__[:]
LOSS_NAMES.append('BCEWithLogitsLoss')


def list_type(s):
    return [int(a) for a in s.split(',')]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='PDS-UKAN')
    parser.add_argument('--epochs', default=500, type=int, metavar='N')
    parser.add_argument('-b', '--batch_size', default=16, type=int, metavar='N')
    parser.add_argument('--dataseed', default=2981, type=int)
    parser.add_argument('--arch', '-a', metavar='ARCH', default='PDSUKAN')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--input_w', default=256, type=int)
    parser.add_argument('--input_h', default=256, type=int)
    parser.add_argument('--input_list', type=list_type, default=[128, 160, 256])
    parser.add_argument('--loss', default='BCEDiceLoss', choices=LOSS_NAMES)
    parser.add_argument('--dataset', default='glas')
    parser.add_argument('--data_dir', default='inputs')
    parser.add_argument('--output_dir', default=r'XXX')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, metavar='LR')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--min_lr', default=1e-5, type=float)

    parser.add_argument('--early_stopping', default=-1, type=int, metavar='N')
    parser.add_argument('--num_workers', default=4, type=int)

    return parser.parse_args()


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {
        'loss': AverageMeter(),
        'iou': AverageMeter(),
    }

    model.train()
    pbar = tqdm(total=len(train_loader))

    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()

        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou, _, _ = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou, _, _ = iou_score(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)

    pbar.close()

    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('iou', avg_meters['iou'].avg),
    ])


def validate(config, val_loader, model, criterion):
    avg_meters = {
        'loss': AverageMeter(),
        'iou': AverageMeter(),
        'dice': AverageMeter(),
    }

    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))

        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou, dice, _ = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou, dice, _ = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)

        pbar.close()

    return OrderedDict([
        ('loss', avg_meters['loss'].avg),
        ('iou', avg_meters['iou'].avg),
        ('dice', avg_meters['dice'].avg),
    ])


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    seed_torch()
    config = vars(parse_args())

    exp_name = config['name']
    save_dir = os.path.join(config['output_dir'], exp_name)
    os.makedirs(save_dir, exist_ok=True)

    print('-' * 20)
    for key in config:
        print(f'{key}: {config[key]}')
    print('-' * 20)

    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()

    cudnn.benchmark = True

    model = PDSUKAN.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision'],
        embed_dims=config['input_list']
    )
    model = model.cuda()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=config['min_lr']
    )

    dataset_name = config['dataset']
    img_ext = '.png'

    if dataset_name == 'busi':
        mask_ext = '_mask.png'
    elif dataset_name == 'glas':
        mask_ext = '.png'
    else:
        raise ValueError(f'Unsupported dataset: {dataset_name}')

    img_ids = sorted(
        glob(os.path.join(config['data_dir'], config['dataset'], 'images', '*' + img_ext))
    )
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    train_img_ids, val_img_ids = train_test_split(
        img_ids,
        test_size=0.2,
        random_state=config['dataseed']
    )

    train_transform = Compose([
        RandomRotate90(),
        geometric.transforms.Flip(),
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join(config['data_dir'], config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=train_transform
    )

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(config['data_dir'], config['dataset'], 'images'),
        mask_dir=os.path.join(config['data_dir'], config['dataset'], 'masks'),
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False
    )

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
    ])

    best_iou = 0
    best_dice = 0
    trigger = 0
    total_train_time = 0.0
    total_val_time = 0.0

    for epoch in range(config['epochs']):
        print(f'Epoch [{epoch}/{config["epochs"]}]')

        start_time = time.time()
        train_log = train(config, train_loader, model, criterion, optimizer)
        total_train_time += time.time() - start_time

        start_time = time.time()
        val_log = validate(config, val_loader, model, criterion)
        total_val_time += time.time() - start_time

        scheduler.step()

        print(
            'loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f'
            % (
                train_log['loss'],
                train_log['iou'],
                val_log['loss'],
                val_log['iou'],
                val_log['dice'],
            )
        )

        current_lr = optimizer.param_groups[0]['lr']

        log['epoch'].append(epoch)
        log['lr'].append(current_lr)
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv(os.path.join(save_dir, 'log.csv'), index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
            best_iou = val_log['iou']
            best_dice = val_log['dice']
            print('=> saved best model')
            print('IoU: %.4f' % best_iou)
            print('Dice: %.4f' % best_dice)
            trigger = 0

        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print('=> early stopping')
            break

        torch.cuda.empty_cache()

    print(f'Total training time: {total_train_time:.2f} seconds')
    print(f'Total validation time: {total_val_time:.2f} seconds')


if __name__ == '__main__':
    main()
