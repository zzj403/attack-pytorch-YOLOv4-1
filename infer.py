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
import json

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
from mmdet.apis.test import *


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('img_file_dir', help='dir of images')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--out_json_name', default='bbox_score.json')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def get_model(args):
    cfg = Config.fromfile(args.config)
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

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_module(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
    return model


def infer(show_score_thr=0.3):
    args = parse_args()
    model = get_model(args)
    img_meta = {'filename': 'temp.jpg',
                'ori_shape': (800, 800, 3),
                'img_shape': (800, 800, 3),
                'pad_shape': (800, 800, 3),
                'scale_factor': np.array([1.000, 1.000, 1.000, 1.000]).astype(np.float32),
                'flip': False,
                'img_norm_cfg': {'mean': np.array([123.675, 116.28, 103.53]).astype(np.float32),
                                 'std': np.array([58.395, 57.12, 57.375]).astype(np.float32),
                                 'to_rgb': True}}
    file_name_list = os.listdir(args.img_file_dir)
    results = {}
    ik = 0
    for file_name in file_name_list:
        if os.path.splitext(file_name)[1] not in ['.jpg', '.png', '.bmp', '.gif']:
            continue
        # ---patched image---
        img_meta['filename'] = file_name
        img = mmcv.imread(args.img_file_dir + file_name)
        img = mmcv.imnormalize(img, img_meta['img_norm_cfg']['mean'], img_meta['img_norm_cfg']['std'])
        img = torch.from_numpy(img).cuda()
        img_shape = (img.size()[0], img.size()[1], img.size()[2])
        img_meta['ori_shape'] = img_shape
        img_meta['img_shape'] = img_shape
        img_meta['pad_shape'] = img_shape
        img = img.permute(2, 0, 1)
        img_p = img.unsqueeze(0)

        # ----clean image----
        img_meta['filename'] = file_name
        img_file_dir2 = args.img_file_dir.replace('_p', '')
        img = mmcv.imread(img_file_dir2 + file_name)
        img = mmcv.imnormalize(img, img_meta['img_norm_cfg']['mean'], img_meta['img_norm_cfg']['std'])
        img = torch.from_numpy(img).cuda()
        img_shape = (img.size()[0], img.size()[1], img.size()[2])
        img_meta['ori_shape'] = img_shape
        img_meta['img_shape'] = img_shape
        img_meta['pad_shape'] = img_shape
        img = img.permute(2, 0, 1)
        img_c = img.unsqueeze(0)

        # pdb.set_trace()
        with torch.no_grad():
            result_p = model(return_loss=False, rescale=True, img=[img_p], img_metas=[[img_meta]])
            result_c = model(return_loss=False, rescale=True, img=[img_c], img_metas=[[img_meta]])
            if isinstance(result_p, tuple):
                bbox_results, mask_results = result_p
                encoded_mask_results = encode_mask_results(mask_results)
                result_p = bbox_results, encoded_mask_results

                bbox_results, mask_results = result_c
                encoded_mask_results = encode_mask_results(mask_results)
                result_c = bbox_results, encoded_mask_results

                # from mmdet.models.detectors.base import
            # model.module.show_result(
            #     img=img_file_dir2 + file_name,
            #     # img_show=True,
            #     result=result_c,
            #     show=True,
            #
            #     score_thr=show_score_thr)

        result_above_confidence_num_p = 0
        result_above_confidence_num_c = 0
        result_p = np.concatenate(result_p)
        result_c = np.concatenate(result_c)
        for ir in range(len(result_p)):
            if result_p[ir, 4] > show_score_thr:
                result_above_confidence_num_p = result_above_confidence_num_p + 1
        for ir in range(len(result_c)):
            if result_c[ir, 4] > show_score_thr:
                result_above_confidence_num_c = result_above_confidence_num_c + 1
        # result_above_confidence = >show_score_thr
        # results[file_name] = np.concatenate(result)


        if result_above_confidence_num_c == 0:  # can't find any object in clean img
            bb_score = 0
            print('i=',ik)
            print(file_name)
            ik += 1
        else:
            bb_score = 1 - min(result_above_confidence_num_c, result_above_confidence_num_p) / result_above_confidence_num_c
        results[file_name] = bb_score

    json_name = args.out_json_name
    with open(json_name, 'w') as f_obj:
        json.dump(results, f_obj)




    return results


if __name__ == '__main__':
    '''python infer.py /disk2/mycode/0511models/mmdetection-master/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py /disk2/mycode/0511models/mmdetection-master/checkpoints/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth ./select1000_p/'''
    results = infer(0.3)
    # json_name = args.out_json_name
    # with open(json_name, 'w') as f_obj:
    #     json.dump(connected_domin_score_dict, f_obj)
    print()

