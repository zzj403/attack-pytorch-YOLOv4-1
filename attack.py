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


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models', default='work-dirs/')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
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
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    parser.add_argument(
        '--clear-work_dir',
        action='store_true',
        help='whether or not to clear the work-dir')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


class PatchTrainer(object):
    def __init__(self, mode):

        self.config = patch_config.patch_configs[mode]()
        self.args = parse_args()

        cfg = Config.fromfile(self.args.config)
        cfg.data.samples_per_gpu = 1
        if self.args.options is not None:
            cfg.merge_from_dict(self.args.options)
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        # work_dir is determined in this priority: CLI > segment in file > filename
        if self.args.work_dir is not None:
            if os.path.exists(self.args.work_dir) is False:
                os.makedirs(self.args.work_dir)
            if self.args.clear_work_dir:
                file_list = os.listdir(self.args.work_dir)
                for f in file_list:
                    if os.path.isdir(os.path.join(self.args.work_dir, f)):
                        shutil.rmtree(os.path.join(self.args.work_dir, f))
                    else:
                        os.remove(os.path.join(self.args.work_dir, f))
            # update configs according to CLI args if args.work_dir is not None
            cfg.work_dir = self.args.work_dir
        elif cfg.get('work_dir', None) is None:
            # use config filename as default work_dir if cfg.work_dir is None
            cfg.work_dir = osp.join('./work_dirs',
                                    osp.splitext(osp.basename(args.config))[0])
        if self.args.resume_from is not None:
            cfg.resume_from = self.args.resume_from
        if self.args.gpu_ids is not None:
            cfg.gpu_ids = self.args.gpu_ids
        else:
            cfg.gpu_ids = range(1) if self.args.gpus is None else range(args.gpus)

        if self.args.autoscale_lr:
            # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
            cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

        # init distributed env first, since logger depends on the dist info.
        if self.args.launcher == 'none':
            distributed = False
        else:
            distributed = True
            init_dist(self.args.launcher, **cfg.dist_params)

        # create work_dir
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        # dump config
        cfg.dump(osp.join(cfg.work_dir, osp.basename(self.args.config)))
        # init the logger before other steps
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

        # init the meta dict to record some important information such as
        # environment info and seed, which will be logged
        meta = dict()
        # log env info
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                    dash_line)
        meta['env_info'] = env_info

        # log some basic info
        logger.info(f'Distributed training: {distributed}')
        logger.info(f'Config:\n{cfg.pretty_text}')

        # set random seeds
        if self.args.seed is not None:
            logger.info(f'Set random seed to {args.seed}, '
                        f'deterministic: {args.deterministic}')
            set_random_seed(self.args.seed, deterministic=args.deterministic)
        cfg.seed = self.args.seed
        meta['seed'] = self.args.seed

        self.model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
        self.model = MMDataParallel(self.model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

        # YOLOv4
        # zzj
        self.darknet_model = Darknet(self.config.cfgfile)
        self.darknet_model.load_weights(self.config.weightfile)
        self.darknet_model = self.darknet_model.eval().cuda()  # TODO: Why eval?


        self.datasets = [build_dataset(cfg.data.train)]
        self.data_loaders = [
            build_dataloader(
                ds,
                cfg.data.samples_per_gpu,
                cfg.data.workers_per_gpu,
                # cfg.gpus will be ignored if distributed
                len(cfg.gpu_ids),
                dist=distributed,
                seed=cfg.seed) for ds in self.datasets
        ]
        if len(cfg.workflow) == 2:
            val_dataset = copy.deepcopy(cfg.data.val)
            val_dataset.pipeline = cfg.data.train.pipeline
            self.datasets.append(build_dataset(val_dataset))
        if cfg.checkpoint_config is not None:
            # save mmdet version, config file content and class names in
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(
                mmdet_version=__version__,
                config=cfg.pretty_text,
                CLASSES=self.datasets[0].CLASSES)
        # add an attribute for visualization convenience
        self.model.CLASSES = self.datasets[0].CLASSES
        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer().cuda()
        self.prob_extractor = MaxProbExtractor(0, 80, self.config).cuda()
        self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.total_variation = TotalVariation().cuda()
        self.writer = self.init_tensorboard(mode)

    def init_tensorboard(self, name=None):
        subprocess.Popen(['tensorboard', '--logdir=runs'])
        if name is not None:
            time_str = time.strftime("%Y%m%d-%H%M%S")
            return SummaryWriter(f'runs/{time_str}_{name}')
        else:
            return SummaryWriter()

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """

        img_size = 800
        img_size = self.darknet_model.height
        n_epochs = 3
        max_batch = 5

        time_str = time.strftime("%Y%m%d-%H%M%S")

        # Generate stating point
        adv_patch_cpu = self.generate_patch("gray")
        #adv_patch_cpu = self.read_image("saved_patches/patchnew0.jpg")

        adv_patch_cpu.requires_grad_(True)

        train_loader = self.data_loaders[0]
        self.epoch_length = len(train_loader)
        print(f'One epoch is {len(train_loader)}')

        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)

        et0 = time.time()
        for epoch in range(n_epochs):
            ep_det_loss = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            bt0 = time.time()
            for i_batch, data in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                        total=self.epoch_length):
                # if i_batch >= max_batch:
                #     break
                # if len(torch.where(data['gt_labels'].data[0][0] == 0)[0]) == 0:
                #     continue
                # else:
                #     person_index = torch.where(data['gt_labels'].data[0][0] == 0)[0]
                #     data['gt_labels'].data[0][0] = data['gt_labels'].data[0][0][person_index]
                #     data['gt_bboxes'].data[0][0] = data['gt_bboxes'].data[0][0][person_index]

                img_mean = data['img_metas'].data[0][0]['img_norm_cfg']['mean']
                img_std = data['img_metas'].data[0][0]['img_norm_cfg']['std']
                img_batch = copy.deepcopy(data['img'].data[0][0])
                img_size_batch = img_batch.size()
                transform = transforms.Compose([
                    transforms.Scale((img_size,
                                      img_size)),
                    transforms.ToTensor(),
                ])
                for channel in range(3):
                    img_batch[channel] = img_batch[channel] * img_std[channel] + img_mean[channel]
                img_batch = torch.where(img_batch < 0, torch.zeros_like(img_batch), img_batch)
                img_batch = torch.where(img_batch > 255, 255 * torch.ones_like(img_batch), img_batch)
                PIL_batch = transforms.ToPILImage()(img_batch.float() / 255.0)
                # PIL_batch.save(self.args.work_dir + str(i_batch) + "_1.jpg")
                img_batch_temp = img_batch.numpy()[[2, 1, 0]].transpose((1, 2, 0)).copy()
                # mmcv.imshow_det_bboxes(img_batch_temp,
                #                        data['gt_bboxes'].data[0][0].numpy(),
                #                        data['gt_labels'].data[0][0].numpy(), class_names=self.model.CLASSES,
                #                        score_thr=0, thickness=1, show=False, wait_time=0,
                #                        out_file=self.args.work_dir + str(i_batch) + "_2.jpg")
                img_batch = transform(PIL_batch).unsqueeze(0)
                # img_batch *= 255.0
                # img_batch_out = img_batch[0][[2, 1, 0]].detach().cpu().float().numpy().transpose((1, 2, 0))
                # mmcv.imwrite(img_batch_out, self.args.work_dir + str(i_batch) + "_3.jpg")
                # for channel in range(3):
                #     img_batch[0][channel] = (img_batch[0][channel] - img_mean[channel]) / img_std[channel]
                lab_batch = data['gt_bboxes'].data[0][0].unsqueeze(0)
                lab_batch[:, :, [0, 2]] /= img_size_batch[2]
                lab_batch[:, :, [1, 3]] /= img_size_batch[1]
                lab_batch[:, :, 2] = lab_batch[:, :, 2] - lab_batch[:, :, 0]
                lab_batch[:, :, 3] = lab_batch[:, :, 3] - lab_batch[:, :, 1]
                lab_batch[:, :, 0] = lab_batch[:, :, 0] + lab_batch[:, :, 2] / 2
                lab_batch[:, :, 1] = lab_batch[:, :, 1] + lab_batch[:, :, 3] / 2

                lab_batch = torch.cat((torch.zeros(1, lab_batch.size()[1], 1), lab_batch), dim=2)
                with autograd.detect_anomaly():
                    img_batch = img_batch.cuda()
                    lab_batch = lab_batch.cuda()
                    #print('TRAINING EPOCH %i, BATCH %i'%(epoch, i_batch))
                    adv_patch = adv_patch_cpu.cuda()
                    adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=False)
                    p_img_batch = self.patch_applier(img_batch, adv_batch_t)
                    p_img_batch = F.interpolate(p_img_batch, (img_size, img_size))

                    # img_origin = data['img'].data[0][0].detach().cpu().float()
                    # batch_img = img_batch[0].detach().cpu().float()
                    # p_img = p_img_batch[0].detach().cpu().float()
                    # for channel in range(3):
                    #     img_origin[channel] = img_origin[channel] * img_std[channel] + img_mean[channel]
                    #     batch_img[channel] = batch_img[channel] * img_std[channel] + img_mean[channel]
                    #     p_img[channel] = p_img[channel] * img_std[channel] + img_mean[channel]
                    # img_origin = img_origin.numpy().transpose((1, 2, 0))[:, :, [2, 1, 0]]
                    # batch_img = batch_img.numpy().transpose((1, 2, 0))[:, :, [2, 1, 0]]
                    # p_img = p_img.numpy().transpose((1, 2, 0))[:, :, [2, 1, 0]]
                    # mmcv.imwrite(img_origin, self.args.work_dir + str(i_batch) + "_0.jpg")
                    # mmcv.imwrite(batch_img, self.args.work_dir + str(i_batch) + "_4.jpg")
                    # mmcv.imwrite(p_img, self.args.work_dir + str(i_batch) + "_5.jpg")

                    # data['img'].data[0][0] = p_img_batch

                    # zzj
                    output = self.darknet_model(p_img_batch)

                    '''
                    output = self.model(**data)
                    det_loss = 0
                    for key in ['loss_rpn_bbox', 'loss_bbox']:
                        if type(output[key]) is list:
                            for losses in output[key]:
                                det_loss += losses
                        else:
                            det_loss += output[key]'''
                    max_prob = self.prob_extractor(output)
                    nps = self.nps_calculator(adv_patch)
                    tv = self.total_variation(adv_patch)

                    det_loss = torch.mean(max_prob)
                    nps_loss = nps * 0.01
                    tv_loss = tv * 2.5
                    loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())

                    ep_det_loss += det_loss.detach().cpu().numpy()
                    ep_nps_loss += nps_loss.detach().cpu().numpy()
                    ep_tv_loss += tv_loss.detach().cpu().numpy()
                    ep_loss += loss

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    adv_patch_cpu.data.clamp_(0,1)       #keep patch in image range

                    bt1 = time.time()
                    if i_batch % 5 == 0:
                        iteration = self.epoch_length * epoch + i_batch

                        self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/nps_loss', nps_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('misc/epoch', epoch, iteration)
                        self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)

                        self.writer.add_image('patch', adv_patch_cpu, iteration)
                    if i_batch + 1 >= len(train_loader):
                        print('\n')
                    else:
                        del adv_batch_t, output, det_loss, p_img_batch, nps_loss, tv_loss, loss
                        torch.cuda.empty_cache()
                    bt0 = time.time()
            et1 = time.time()
            ep_det_loss = ep_det_loss/len(train_loader)
            ep_nps_loss = ep_nps_loss/len(train_loader)
            ep_tv_loss = ep_tv_loss/len(train_loader)
            ep_loss = ep_loss/len(train_loader)

            #im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            #plt.imshow(im)
            #plt.savefig(f'pics/{time_str}_{self.config.patch_name}_{epoch}.png')

            scheduler.step(ep_loss)
            if True:
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss)
                print('  DET LOSS: ', ep_det_loss)
                print('  NPS LOSS: ', ep_nps_loss)
                print('   TV LOSS: ', ep_tv_loss)
                print('EPOCH TIME: ', et1-et0)
                #im = transforms.ToPILImage('RGB')(adv_patch_cpu)
                #plt.imshow(im)
                #plt.show()
                #im.save("saved_patches/patchnew1.jpg")
                # del adv_batch_t, output, det_loss, p_img_batch, nps_loss, tv_loss, loss
                torch.cuda.empty_cache()
            et0 = time.time()

    def generate_patch(self, type):
        """
        Generate a random patch as a starting point for optimization.

        :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        :return:
        """
        if type == 'gray':
            adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)
        elif type == 'random':
            adv_patch_cpu = torch.rand((3, self.config.patch_size, self.config.patch_size))

        return adv_patch_cpu

    def read_image(self, path):
        """
        Read an input image to be used as a patch

        :param path: Path to the image to be read.
        :return: Returns the transformed patch as a pytorch Tensor.
        """
        patch_img = Image.open(path).convert('RGB')
        tf = transforms.Resize((self.config.patch_size, self.config.patch_size))
        patch_img = tf(patch_img)
        tf = transforms.ToTensor()

        adv_patch_cpu = tf(patch_img)
        return adv_patch_cpu


if __name__ == '__main__':
    Trainer = PatchTrainer('base')
    Trainer.train()
