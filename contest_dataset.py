import argparse
import copy
import os
import os.path as osp
import time
from load_data import *
import gc
import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
from tensorboardX import SummaryWriter
import subprocess
import patch_config
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import init_dist
import random
from collections import OrderedDict
import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import DistSamplerSeedHook, Runner
import PIL
import load_data
from tqdm import tqdm
import pdb
import copy
import cv2
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
import shutil

import sys
sys.path.append('../mmdetection-master/')
from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
from mmdet.core import (DistEvalHook, DistOptimizerHook, EvalHook,
                        Fp16OptimizerHook, build_optimizer)
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.core import tensor2imgs


def parse_args():
    parser = argparse.ArgumentParser(
        description='Dataset generator')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'save_dir', help='directory where images will be saved')
    parser.add_argument('--seed', type=int, default=1)
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument(
        '--clear-save_dir',
        action='store_true',
        help='whether or not to clear the work-dir')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    set_random_seed(args.seed, deterministic=True)
    if args.save_dir is not None:
        if os.path.exists(args.save_dir) is False:
            os.makedirs(args.save_dir)
        if args.clear_save_dir:
            file_list = os.listdir(args.save_dir)
            for f in file_list:
                if os.path.isdir(os.path.join(args.save_dir, f)):
                    shutil.rmtree(os.path.join(args.save_dir, f))
                else:
                    os.remove(os.path.join(args.save_dir, f))
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    data_length = 2000
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.seed = args.seed
    cfg.data.samples_per_gpu = 2
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    datasets = [build_dataset(cfg.data.train)]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed) for ds in datasets
    ]
    data_loader = data_loaders[0]
    output = []
    ia = 0
    for i_batch, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        if len(output) >= data_length:
            break
        imgs = data['img'].data[0]
        img_metas = data['img_metas'].data[0]
        img_tensor = imgs.detach()
        imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
        for i in range(len(imgs)):
            h, w, _ = img_metas[i]['img_shape']
            img = imgs[i][:h, :w, :]
            gt_bboxes = data['gt_bboxes'].data[0][i]
            area = torch.mul((gt_bboxes[:, 2] - gt_bboxes[:, 0]), (gt_bboxes[:, 3] - gt_bboxes[:, 1]))
            if torch.min(area / (h * w)) < 0.05:
                continue
            if 4 <= gt_bboxes.size()[0] <= 15:

                # output.append(cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_CUBIC))
                img_o = cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_CUBIC)
                mmcv.imwrite(img_o, args.save_dir + '/' + str(ia) + '.png')
                print(ia)
                ia += 1
    print('finish')



if __name__ == '__main__':
    main()
