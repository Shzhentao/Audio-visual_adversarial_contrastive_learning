# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import time
import yaml
import torch
import math
import os
from utils import eval_utils
import utils.logger
import torch.multiprocessing as mp
import shutil


parser = argparse.ArgumentParser(description='Evaluation on ESC Sound Classification')
parser.add_argument('cfg', metavar='CFG', help='config file')
parser.add_argument('model_cfg', metavar='CFG', help='config file')
parser.add_argument('--quiet', action='store_true')
parser.add_argument('--test_only', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--port', default='1234')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
parser.add_argument('--avc', action='store_true', help='scratch training')
parser.add_argument('--scratch', action='store_true', help='scratch training')
parser.add_argument('--ft_all', action='store_true', help='scratch training')
parser.add_argument('--sample_grad_cam_audio', action='store_true', help='sample grad cam audio')


def main():
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    if args.test_only:
        cfg['test_only'] = True
    if args.resume:
        cfg['resume'] = True
    if args.ft_all:
        cfg['model']['args']['ft_all'] = True
    if args.debug:
        cfg['num_workers'] = 1
        cfg['dataset']['batch_size'] = 4
    if args.sample_grad_cam_audio:
        cfg['dataset']['sample_grad_cam_audio'] = True

    ngpus = torch.cuda.device_count()
    if args.distributed:
        mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, cfg['dataset']['fold'], args, cfg))
    else:
        main_worker(None, ngpus, cfg['dataset']['fold'], args, cfg)


def main_worker(gpu, ngpus, fold, args, cfg):
    args.gpu = gpu
    args.world_size = ngpus
    if args.sample_grad_cam_audio and args.gpu == 0:
        data_address = './data/esc'
        if not os.path.exists(data_address):
            os.makedirs(data_address)
        else:
            shutil.rmtree(data_address)
            os.makedirs(data_address)
        args.data_address = data_address

    # Prepare folder and logger
    eval_dir, model_cfg, logger = eval_utils.prepare_environment(args, cfg, fold)

    # Model
    model, ckp_manager = eval_utils.build_model(model_cfg, cfg, eval_dir, args, logger)

    # Optimizer
    optimizer = eval_utils.build_optimizer(model.parameters(), cfg['optimizer'], logger)

    # Datasets
    train_loader, test_loader, dense_loader = eval_utils.build_dataloaders(
        cfg['dataset'], fold, cfg['num_workers'], args.distributed, logger)

    # ################################ Train ############################### #
    start_epoch, end_epoch = 0, cfg['optimizer']['num_epochs']
    if cfg['resume'] or args.test_only:
        if ckp_manager.checkpoint_exists(last=True):
            start_epoch = ckp_manager.restore(model, optimizer, restore_last=True)
            logger.add_line("Loaded checkpoint '{}' (epoch {})".format(ckp_manager.last_checkpoint_fn(), start_epoch - 1))
        else:
            raise AttributeError

    if not cfg['test_only']:
        logger.add_line("=" * 30 + "   Training   " + "=" * 30)
        gamma = cfg['optimizer']['lr']['gamma']
        num_file = 0
        while os.path.exists('{}/model_best{}.pth.tar'.format(eval_dir, num_file)):
            num_file += 1
        for epoch in range(start_epoch, end_epoch):
            # ckp_manager.save(model, optimizer, epoch, eval_metric=1, num_file=num_file)
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                test_loader.sampler.set_epoch(epoch)
            adjust_learning_rate(optimizer, cfg['optimizer']['lr']['base_lr'],
                                 epoch, end_epoch, gamma, cfg['optimizer']['lr']['milestones'], args)

            logger.add_line('='*30 + ' Epoch {} '.format(epoch) + '='*30)
            logger.add_line('LR: {}'.format(get_learning_rate(optimizer)))
            # print('=' * 30 + ' Epoch {} '.format(epoch) + '=' * 30)
            # print('LR: {}'.format(get_learning_rate(optimizer)))
            if args.sample_grad_cam_audio:
                run_phase('test', test_loader, model, None, epoch, args, cfg, logger)
                break
            run_phase('train', train_loader, model, optimizer, epoch, args, cfg, logger)
            top1, _ = run_phase('test', test_loader, model, None, epoch, args, cfg, logger)
            # for ft in top1:
            #     top1_val = top1[ft]
            ckp_manager.save(model, optimizer, epoch, eval_metric=top1, num_file=num_file)
            save_freq = cfg['optimizer']['lr']['save_freq']
            if (epoch+1) % save_freq == 0:
                ckp_manager.save(model, optimizer, epoch, filename='checkpoint-ep{}.pth.tar'.format(epoch),num_file=num_file)

    # ############################### Eval ############################### #
    if not args.sample_grad_cam_audio:
        logger.add_line('\n' + '=' * 30 + ' Final evaluation ' + '=' * 30)
        cfg['dataset']['test']['clips_per_video'] = 25
        train_loader, test_loader, dense_loader = eval_utils.build_dataloaders(cfg['dataset'],
                                                                            fold, cfg['num_workers'],
                                                                            args.distributed, logger)
        top1, top5 = run_phase('test', test_loader, model, None, end_epoch, args, cfg, logger)
        top1_dense, top5_dense = run_phase('test_dense', dense_loader, model, None, end_epoch, args, cfg, logger)

        logger.add_line('\n' + '=' * 30 + ' Evaluation done ' + '=' * 30)
        logger.add_line('Clip@1: {:6.2f}'.format(top1))
        logger.add_line('Clip@5: {:6.2f}'.format(top5))
        logger.add_line('Video@1: {:6.2f}'.format(top1_dense))
        logger.add_line('Video@5: {:6.2f}'.format(top5_dense))
        # for ft in top1:
            # logger.add_line('')
            # logger.add_line('[{}] Clip@1: {:6.2f}'.format(ft, top1[ft]))
            # logger.add_line('[{}] Clip@5: {:6.2f}'.format(ft, top5[ft]))
            # logger.add_line('[{}] Video@1: {:6.2f}'.format(ft, top1_dense[ft]))
            # logger.add_line('[{}] Video@5: {:6.2f}'.format(ft, top5_dense[ft]))


def run_phase(phase, loader, model, optimizer, epoch, args, cfg, logger):
    from utils import metrics_utils
    logger.add_line('\n{}: Epoch {}'.format(phase, epoch))
    # feature_names = cfg['model']['args']['feat_names']
    batch_time = metrics_utils.AverageMeter('Time', ':6.6f', window_size=100)
    data_time = metrics_utils.AverageMeter('Data', ':6.6f', window_size=100)
    loss_meters = metrics_utils.AverageMeter('Loss', ':.4e')
    top1_meters = metrics_utils.AverageMeter('Acc@1', ':6.6f')
    top5_meters = metrics_utils.AverageMeter('Acc@5', ':6.6f')
    # loss_meters = {ft: metrics_utils.AverageMeter('Loss', ':.4e', 0) for ft in feature_names}
    # top1_meters = {ft: metrics_utils.AverageMeter('Acc@1', ':6.2f', 0) for ft in feature_names}
    # top5_meters = {ft: metrics_utils.AverageMeter('Acc@5', ':6.2f', 0) for ft in feature_names}
    # progress = {'timers': utils.logger.ProgressMeter(len(loader),
    #                                                  meters=[batch_time, data_time],
    #                                                  phase=phase, epoch=epoch, logger=logger)}
    # progress.update({ft: utils.logger.ProgressMeter(len(loader),
                                                    # meters=[loss_meters[ft], top1_meters[ft], top5_meters[ft]],
                                                    # phase=phase, epoch=epoch, logger=logger) for ft in feature_names})
    progress = utils.logger.ProgressMeter(len(loader),
                                          meters=[batch_time, data_time, loss_meters, top1_meters, top5_meters],
                                          phase=phase, epoch=epoch, logger=logger)

    # switch to train/test mode
    model.train(phase == 'train')

    if phase in {'test_dense', 'test'}:
        model = BatchWrapper(model, cfg['dataset']['batch_size'])

    end = time.time()
    criterion = torch.nn.MultiMarginLoss()
    softmax = torch.nn.Softmax(dim=1)
    for it, sample in enumerate(loader):
        data_time.update(time.time() - end)

        audio = sample['audio0']
        # print(audio.size())  # torch.Size([32, 1, 200, 257])
        # print(type(audio))  # <class 'torch.Tensor'>
        # print(audio.dtype)  # torch.float32
        if it == 0 and args.gpu == 0 and args.sample_grad_cam_audio and not phase == 'test_dense':
            audio_origin = sample['audio2']
            # print(audio_origin.shape)  # torch.Size([32, 1, 48000])
            # print(type(audio_origin))  # <class 'torch.Tensor'>
            # print(audio_origin.dtype)  # torch.float64
            torch.save(audio_origin, "{}/audio-origin-{:02d}-{:05d}.pt".format(args.data_address, args.gpu, it))
            torch.save(audio, "{}/audio{:02d}-{:05d}.pt".format(args.data_address, args.gpu, it))
        if args.sample_grad_cam_audio:
            break
        target = sample['label'].cuda()

        if args.gpu is not None:
            audio = audio.cuda(args.gpu, non_blocking=True)
        if torch.cuda.device_count() == 1 and args.gpu is None:
            audio = audio.cuda()

        if phase == 'test_dense':
            batch_size, clips_per_sample = audio.shape[0], audio.shape[1]
            # flattens a contiguous range of dims[0, 1]
            audio = audio.flatten(0, 1).contiguous()

        # compute outputs
        # weights = model.module.classifiers.classifier.weight
        # weights = {}
        # for cla, ft in zip(model.module.classifiers, cfg['model']['args']['feat_names']):
        #     weights.update({ft: cla.classifier.weight})
        if phase == 'train':
            logits = model(audio)
        else:
            with torch.no_grad():
                logits = model(audio)

        # compute loss and measure accuracy
        total_loss = 0.
        # flag_ft = 0
        param_c = cfg['model']['c']

        # for ft in feature_names:
        with torch.no_grad():
            # input_dim = cfg['model']['args']['feat_dims'][flag_ft]
            input_dim = cfg['model']['args']['feat_dims']
            fenm = input_dim * input_dim
        if phase == 'test_dense':
            # confidence = logits[ft].view(batch_size, clips_per_sample, -1).mean(1)
            # confidence = logits.view(batch_size, clips_per_sample, -1).mean(1)
            target_tiled = target.unsqueeze(1).repeat(1, clips_per_sample).view(-1)
            # loss = criterion(logits[ft], target_tiled)
            loss = criterion(logits, target_tiled)
            # loss += param_c * torch.pow(torch.norm(weights[ft]), 2) / fenm
            # loss += param_c * torch.pow(torch.norm(weights), 2) / fenm
        else:
            # confidence = softmax(logits[ft])
            # loss = criterion(logits[ft], target)
            loss = criterion(logits, target)
            # loss += param_c * torch.pow(torch.norm(weights[ft]), 2) / fenm
            # loss += param_c * torch.pow(torch.norm(weights), 2) / fenm
        # total_loss += loss
        # flag_ft += 1

        with torch.no_grad():
            if phase == 'test_dense':
                # confidence = logits.view(batch_size, clips_per_sample, -1).mean(1)
                # confidence = softmax(logits)
                confidence = softmax(logits).view(batch_size, clips_per_sample, -1).mean(1)
            else:
                confidence = softmax(logits)
            acc1, acc5 = metrics_utils.accuracy(confidence, target, topk=(1, 5))
            loss_meters.update(loss.item(), target.size(0))
            top1_meters.update(acc1[0].item(), target.size(0))
            top5_meters.update(acc5[0].item(), target.size(0))

        # compute gradient and do SGD step
        if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (it + 1) % 100 == 0 or it == 0 or it + 1 == len(loader):
            progress.display(it+1, pr2term=False)
            # for ft in progress:
            #     progress[ft].display(it+1, pr2term=False)

    if args.distributed and not args.sample_grad_cam_audio:
        progress.synchronize_meters(args.gpu)
        progress.display(len(loader) * args.world_size, pr2term=False)
        # for ft in progress:
        #     progress[ft].synchronize_meters(args.gpu)
        #     progress[ft].display(len(loader) * args.world_size, pr2term=False)

        # return {ft: top1_meters[ft].avg for ft in feature_names}, {ft: top5_meters[ft].avg for ft in feature_names}
    if not args.sample_grad_cam_audio:
        return top1_meters.avg, top5_meters.avg


# class BatchWrapper:
#     def __init__(self, model, batch_size, cfg):
#         self.model = model
#         self.batch_size = batch_size
#         self.cfg = cfg

#     def __call__(self, x):
#         from collections import defaultdict
#         outs = defaultdict(list)
#         for i in range(0, x.shape[0], self.batch_size):
#             odict = self.model(x[i:i + self.batch_size])
#             for k in odict:
#                 outs[k] += [odict[k]]
#         for k in outs:
#             outs[k] = torch.cat(outs[k], 0)
#         # weights = {}
#         # for cla, ft in zip(self.model.module.classifiers, self.cfg['model']['args']['feat_names']):
#         #     # print(cla.classifier.weight.size())
#         #     weights.update({ft: cla.classifier.weight})
#         # # print(weights)
#         return outs


class BatchWrapper:
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

    def __call__(self, x):
        outs = []
        for i in range(0, x.shape[0], self.batch_size):
            outs += [self.model(x[i:i + self.batch_size])]  # torch.Size([64, 34])  # torch.Size([32, 34])
        return torch.cat(outs, 0)


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
