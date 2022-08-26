# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import random
import time
import warnings
import yaml
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp

import utils.logger
from utils import main_utils
import 

parser = argparse.ArgumentParser(description='PyTorch AVAC Training')

parser.add_argument('cfg', help='model directory')
parser.add_argument('--quiet', action='store_true')

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:15475', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
parser.add_argument('--use_shuffle', action='store_true', help='use cosine lr schedule')
parser.add_argument('--resume', action='store_true', help='use cosine lr schedule')
parser.add_argument('--simple_model', action='store_true', help='use cosine lr schedule')


def main():
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    cfg['model']['args']['use_shuffle'] = args.use_shuffle
    cfg['resume'] = args.resume
    cfg['loss']['args']['simple_model'] = args.simple_model
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, cfg))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, cfg)


def main_worker(gpu, ngpus_per_node, args, cfg):
    args.gpu = gpu
    # Setup environment
    args = main_utils.initialize_distributed_backend(args, ngpus_per_node)
    logger, tb_writter, model_dir = main_utils.prep_environment(args, cfg)

    # Define model and cross_criterion
    base_model = main_utils.build_model(cfg['model'], logger)
    # device = args.gpu if args.gpu is not None else 0
    cfg['loss']['args']['embedding_dim'] = base_model.out_dim
    # define loss function (criterion)
    avac_criterion = main_utils.build_avac_criterion(cfg['loss'], logger=logger)
    models = [base_model, avac_criterion]
    models, args, cfg['dataset']['batch_size'], cfg['num_workers'] = main_utils.distribute_model_to_cuda(
        models, args, cfg['dataset']['batch_size'], cfg['num_workers'], ngpus_per_node)
    base_model, avac_criterion = models[0], models[1]

    # Define dataloaders
    train_loader = main_utils.build_dataloaders(cfg['dataset'], cfg['num_workers'], args.distributed, logger)

    # Define optimizer
    optimizers = main_utils.build_optimizer(
        params_base_model=list(base_model.parameters()),
        params_avac_criterion=list(avac_criterion.parameters()),
        cfg=cfg['optimizer'],
        logger=logger)
    optimizer_base_model, optimizer_neg = optimizers[0], optimizers[1]
    # scheduler_base_model, scheduler_neg = schedulers[0], schedulers[1]

    ckp_manager = main_utils.CheckpointManager(model_dir, rank=args.rank)

    # Optionally resume from a checkpoint
    start_epoch, end_epoch = 0, cfg['optimizer']['num_epochs']
    gamma = cfg['optimizer']['lr']['gamma']
    if cfg['resume']:
        if ckp_manager.checkpoint_exists(last=True):
            start_epoch = ckp_manager.restore(restore_last=True, base_model=base_model, avac_criterion=avac_criterion,
                                              optimizer_base_model=optimizer_base_model,
                                              optimizer_neg=optimizer_neg)
            logger.add_line("Checkpoint loaded: '{}' (epoch {})".format(ckp_manager.last_checkpoint_fn(), start_epoch-1))
        else:
            logger.add_line("No checkpoint found at '{}'".format(ckp_manager.last_checkpoint_fn()))
            raise AttributeError

    cudnn.benchmark = True

    #  ########################### TRAIN #########################################  #
    test_freq = cfg['test_freq'] if 'test_freq' in cfg else 1
    for epoch in range(start_epoch, end_epoch):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer_base_model, cfg['optimizer']['lr']['base_lr'],
                             epoch, end_epoch, gamma, cfg['optimizer']['lr']['milestones'], args)
        adjust_learning_rate(optimizer_neg, cfg['optimizer']['lr']['negative_lr'],
                             epoch, end_epoch, gamma, cfg['optimizer']['lr']['milestones'], args)

        # Train for one epoch
        logger.add_line('=' * 30 + ' Epoch {} '.format(epoch) + '=' * 30)
        logger.add_line('model LR: {}'.format(get_learning_rate(optimizer_base_model)))
        logger.add_line('cross LR: {}'.format(get_learning_rate(optimizer_neg)))
        run_phase('train', train_loader, base_model, avac_criterion, optimizer_base_model,
                  optimizer_neg, epoch, args, cfg, logger, tb_writter)
        if epoch % test_freq == 0 or epoch == end_epoch - 1:
            ckp_manager.save(epoch + 1, base_model=base_model, avac_criterion=avac_criterion,
                             optimizer_base_model=optimizer_base_model, optimizer_neg=optimizer_neg)
        save_freq = cfg['optimizer']['lr']['save_freq']
        if (epoch+1) % save_freq == 0:
            ckp_manager.save(epoch + 1, base_model=base_model, avac_criterion=avac_criterion,
                             optimizer_base_model=optimizer_base_model, optimizer_neg=optimizer_neg,
                             filename='checkpoint-ep{}.pth.tar'.format(epoch))


def run_phase(phase, loader, base_model, avac_criterion, optimizer_base_model, optimizer_neg,
              epoch, args, cfg, logger, tb_writter):
    from utils import metrics_utils
    logger.add_line('\n{}: Epoch {}'.format(phase, epoch))
    data_time = metrics_utils.AverageMeter('Data', ':6.6f', window_size=100)
    batch_time = metrics_utils.AverageMeter('Time', ':6.6f', window_size=100)
    loss_base_meter = metrics_utils.AverageMeter('Loss', ':.6e')
    loss_neg_meter = metrics_utils.AverageMeter('Loss-x', ':.6e')
    progress = utils.logger.ProgressMeter(len(loader), [data_time, batch_time, loss_base_meter, loss_neg_meter],
                                          phase=phase, epoch=epoch, logger=logger, tb_writter=tb_writter)

    # switch to train mode
    base_model.train(phase == 'train')

    end = time.time()
    device = args.gpu if args.gpu is not None else 0
    for i, sample in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # Prepare batch
        video_q, video_k, audio_q, audio_k, index = sample['frames0'], sample['frames1'], sample['audio0'], sample[
            'audio1'], sample['index']
        # audio_origin = sample['audio2']
        # print(audio_origin.size())
        video_q = video_q.cuda(device, non_blocking=True)
        video_k = video_k.cuda(device, non_blocking=True)
        audio_q = audio_q.cuda(device, non_blocking=True)
        audio_k = audio_k.cuda(device, non_blocking=True)
        index = index.cuda(device, non_blocking=True)
        # print(video_q.size())  # torch.Size([48, 3, 8, 224, 224])
        # print(audio_q.size())  # torch.Size([48, 1, 200, 257])

        # compute audio and video embeddings
        if phase == 'train':
            video_emb_q, video_emb_k, audio_emb_q, audio_emb_k = base_model(video_q, video_k, audio_q, audio_k)
            # print(video_emb_k.size())  torch.Size([48, 32])
        else:
            with torch.no_grad():
                video_emb_q, video_emb_k, audio_emb_q, audio_emb_k = base_model(video_q, video_k, audio_q, audio_k)

        # if phase_subset == 'base':

        # compute base loss
        # loss_base = avac_criterion(video_emb_q, video_emb_k,
        #                            audio_emb_q, audio_emb_k, index, update_memory=False)
        # with torch.no_grad():
        #     loss_base_meter.update(loss_base.item(), video_q.size(0))
        #
        # # compute gradient and do SGD step during training
        # if phase == 'train':
        #     optimizer_base_model.zero_grad()
        #     loss_base.backward(retain_graph=True)
        #     optimizer_base_model.step()
        # # elif phase_subset == 'adversarial':
        #
        # # compute avac loss
        # loss_neg = avac_criterion(video_emb_q.clone().detach(), video_emb_k.clone().detach(),
        #                           audio_emb_q, audio_emb_k, index, update_memory=True)
        # with torch.no_grad():
        #     loss_neg_meter.update(-loss_neg.item(), video_q.size(0))
        #
        # # compute gradient and do SGD step during training
        # if phase == 'train':
        #     optimizer_neg.zero_grad()
        #     loss_neg.backward()
        #     optimizer_neg.step()
        # loss_base = avac_criterion(video_emb_q, video_emb_k,
        #                            audio_emb_q, audio_emb_k, index, update_memory=False)
        # with torch.no_grad():
        #     loss_base_meter.update(loss_base.item(), video_q.size(0))
        #
        # # compute gradient and do SGD step during training
        # if phase == 'train':
        #     optimizer_base_model.zero_grad()
        #     loss_base.backward(retain_graph=True)
        #     optimizer_base_model.step()
        # # elif phase_subset == 'adversarial':
        #
        # # compute avac loss
        # loss_neg = avac_criterion(video_emb_q.clone().detach(), video_emb_k.clone().detach(),
        #                           audio_emb_q, audio_emb_k, index, update_memory=True)
        # with torch.no_grad():
        #     loss_neg_meter.update(-loss_neg.item(), video_q.size(0))
        #
        # # compute gradient and do SGD step during training
        # if phase == 'train':
        #     optimizer_neg.zero_grad()
        #     loss_neg.backward()
        #     optimizer_neg.step()

        loss_base = avac_criterion(video_emb_q, video_emb_k,
                                   audio_emb_q, audio_emb_k, index, update_memory=False)
        with torch.no_grad():
            loss_base_meter.update(loss_base.item(), video_q.size(0))

        # compute gradient and do SGD step during training
        if phase == 'train':
            optimizer_base_model.zero_grad()
            loss_base.backward(retain_graph=True)
        # elif phase_subset == 'adversarial':

        # compute avac loss
        loss_neg = avac_criterion(video_emb_q.clone().detach(), video_emb_k.clone().detach(),
                                  audio_emb_q, audio_emb_k, index, update_memory=True)
        with torch.no_grad():
            loss_neg_meter.update(-loss_neg.item(), video_q.size(0))

        # compute gradient and do SGD step during training
        if phase == 'train':
            optimizer_neg.zero_grad()
            loss_neg.backward()
            optimizer_base_model.step()
            optimizer_neg.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print to terminal and tensorboard
        # step = epoch * len(loader) + i
        if (i + 1) % cfg['print_freq'] == 0 or i == 0 or i + 1 == len(loader):
            progress.display(i + 1)
            # if tb_writter is not None:
            #     for key in loss_debug:
            #         tb_writter.add_scalar('{}-batch/{}'.format(phase, key), loss_debug[key].item(), step)

    # Sync metrics across all GPUs and print final averages
    if args.distributed:
        progress.synchronize_meters(args.gpu)
        progress.display(len(loader) * args.world_size, pr2tb=False)

    if tb_writter is not None:
        for meter in progress.meters:
            tb_writter.add_scalar('{}-epoch/{}'.format(phase, meter.name), meter.avg, epoch)


def adjust_learning_rate(optimizer, origin_lr, epoch, num_epochs, gamma, milestones, args):
    """Decay the learning rate based on schedule"""
    lr = origin_lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / num_epochs))
    else:  # stepwise lr schedule
        for milestone in milestones:
            lr *= gamma if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr = 0
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    return lr


if __name__ == '__main__':
    main()
