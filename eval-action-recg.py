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
import utils.logger
from utils import eval_utils
import torch.multiprocessing as mp
import shutil


parser = argparse.ArgumentParser(description='Evaluation on ESC Sound Classification')
parser.add_argument('cfg', metavar='CFG', help='config file')
parser.add_argument('model_cfg', metavar='CFG', help='config file')
parser.add_argument('--quiet', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--test-only', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--port', default='1234')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
parser.add_argument('--scratch', action='store_true', help='scratch training')
parser.add_argument('--avc', action='store_true', help='avc')
parser.add_argument('--sample_grad_cam', action='store_true', help='sample grad cam')
parser.add_argument('--adco', action='store_true', help='sample grad cam')


def main():
    ngpus = torch.cuda.device_count()
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    if args.test_only:
        cfg['test_only'] = True
    if args.resume:
        cfg['resume'] = True
    if args.debug:
        cfg['num_workers'] = 1
        cfg['dataset']['batch_size'] = 4
    if args.sample_grad_cam:
        cfg['dataset']['sample_grad_cam'] = True
        cfg['num_workers'] = 1
        cfg['dataset']['batch_size'] = 1
        data_addresses = ['./data/ucf/visual', './data/ucf/visual/heatmap', './data/ucf/visual_scratch', './data/ucf/visual_scratch/heatmap_scratch']
        for data_address in data_addresses:
            if not os.path.exists(data_address):
                os.makedirs(data_address)
            else:
                shutil.rmtree(data_address)
                os.makedirs(data_address)
        args.data_address = data_addresses[0]
        args.data_address_heatmap = data_addresses[1]
        args.data_address_scratch = data_addresses[2]
        args.data_address_heatmap_scratch = data_addresses[3]

    if args.distributed:
        mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, cfg['dataset']['fold'], args, cfg))
    else:
        main_worker(None, ngpus, cfg['dataset']['fold'], args, cfg)


def main_worker(gpu, ngpus, fold, args, cfg):
    args.gpu = gpu
    args.world_size = ngpus

    # Prepare folder and logger
    eval_dir, model_cfg, logger = eval_utils.prepare_environment(args, cfg, fold)

    # Model
    if not args.sample_grad_cam:
        model, ckp_manager = eval_utils.build_model(model_cfg, cfg, eval_dir, args, logger)
    else:
        model1, model2, ckp_manager = eval_utils.build_model(model_cfg, cfg, eval_dir, args, logger)
        model = [model1, model2]

    # Optimizer
    if not args.sample_grad_cam:
        optimizer = eval_utils.build_optimizer(model.parameters(), cfg['optimizer'], logger)
    else:
        optimizer = None

    # Datasets
    train_loader, test_loader, dense_loader = eval_utils.build_dataloaders(
        cfg['dataset'], fold, cfg['num_workers'], args.distributed, logger)

    # ############################### Train ############################### #
    start_epoch, end_epoch = 0, cfg['optimizer']['num_epochs']
    if cfg['resume'] or args.test_only:
        if ckp_manager.checkpoint_exists(last=True):
            start_epoch = ckp_manager.restore(model, optimizer, restore_last=True)
            logger.add_line("Loaded checkpoint '{}' (epoch {})".format(ckp_manager.last_checkpoint_fn(), start_epoch - 1))
        else:
            raise AttributeError

    if not cfg['test_only']:
        logger.add_line("=" * 30 + "   Training   " + "=" * 30)

        # Warmup. Train classifier for a few epochs.
        if start_epoch == 0 and 'warmup_classifier' in cfg['optimizer'] and cfg['optimizer']['warmup_classifier'] and not args.sample_grad_cam:
            n_wu_epochs = cfg['optimizer']['warmup_epochs'] if 'warmup_epochs' in cfg['optimizer'] else 5
            cls_opt = eval_utils.build_optimizer(
                params=[p for n, p in model.named_parameters() if 'feature_extractor' not in n],
                cfg={'lr': {'base_lr': cfg['optimizer']['lr']['base_lr'], 'milestones': [n_wu_epochs], 'gamma': 1.},
                     'weight_decay': cfg['optimizer']['weight_decay'],
                     'name': cfg['optimizer']['name']}
            )
            for epoch in range(n_wu_epochs):
                run_phase('train', train_loader, model, cls_opt, epoch, args, cfg, logger)
                top1, _ = run_phase('test', test_loader, model, None, epoch, args, cfg, logger)

        gamma = cfg['optimizer']['lr']['gamma']
        num_file = 0
        while os.path.exists('{}/model_best{}.pth.tar'.format(eval_dir, num_file)):
            num_file += 1
        # Main training loop
        for epoch in range(start_epoch, end_epoch):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                test_loader.sampler.set_epoch(epoch)
            if args.sample_grad_cam:
                run_phase('train', train_loader, model, optimizer, epoch, args, cfg, logger)
                break
            adjust_learning_rate(optimizer, cfg['optimizer']['lr']['base_lr'],
                                 epoch, end_epoch, gamma, cfg['optimizer']['lr']['milestones'], args)

            logger.add_line('='*30 + ' Epoch {} '.format(epoch) + '='*30)
            logger.add_line('LR: {}'.format(get_learning_rate(optimizer)))
            # print('='*30 + ' Epoch {} '.format(epoch) + '='*30)
            # print('LR: {}'.format(get_learning_rate(optimizer)))
            run_phase('train', train_loader, model, optimizer, epoch, args, cfg, logger)
            top1, _ = run_phase('test', test_loader, model, None, epoch, args, cfg, logger)
            ckp_manager.save(model, optimizer, epoch, eval_metric=top1,num_file=num_file)
            save_freq = cfg['optimizer']['lr']['save_freq']
            if (epoch+1) % save_freq == 0:
                ckp_manager.save(model, optimizer, epoch, filename='checkpoint-ep{}.pth.tar'.format(epoch),num_file=num_file)

    # ############################### Eval ############################### #
    if not args.sample_grad_cam:
        logger.add_line('\n' + '=' * 30 + ' Final evaluation ' + '=' * 30)
        # Evaluate clip-level predictions with 25 clips per video for metric stability
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


def run_phase(phase, loader, model, optimizer, epoch, args, cfg, logger):
    from utils import metrics_utils
    batch_time = metrics_utils.AverageMeter('Time', ':6.6f', window_size=100)
    data_time = metrics_utils.AverageMeter('Data', ':6.6f', window_size=100)
    loss_meter = metrics_utils.AverageMeter('Loss', ':.4e')
    top1_meter = metrics_utils.AverageMeter('Acc@1', ':6.6f')
    top5_meter = metrics_utils.AverageMeter('Acc@5', ':6.6f')
    progress = utils.logger.ProgressMeter(len(loader),
                                          meters=[batch_time, data_time, loss_meter, top1_meter, top5_meter],
                                          phase=phase, epoch=epoch, logger=logger)

    # switch to train/test mode
    if args.sample_grad_cam:
        model2 = model[1]
        model = model[0]
        model2.train(phase == 'train')
    model.train(phase == 'train')
    if phase in {'test_dense', 'test'}:
        model = eval_utils.BatchWrapper(model, cfg['dataset']['batch_size'])

    criterion = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)

    end = time.time()
    logger.add_line('\n{}: Epoch {}'.format(phase, epoch))
    # print('\n{}: Epoch {}'.format(phase, epoch))
    for it, sample in enumerate(loader):
        data_time.update(time.time() - end)
        video = sample['frames0']
        target = sample['label']
        # print(video.size())  # torch.Size([32, 3, 8, 224, 224])
        # print(video)

        if args.sample_grad_cam and not phase == 'test_dense':
            video_origin = sample['frames2']
            torch.save(video_origin, "{}/video-origin-{:02d}-{:07d}.pt".format(args.data_address, args.gpu, it))
            torch.save(video, "{}/video{:02d}-{:07d}.pt".format(args.data_address, args.gpu, it))
        
        # index = sample['index'].cuda()
        # print(video.size())  # torch.Size([32, 3, 8, 224, 224])
        # print(target.size())  # torch.Size([32])

        if args.gpu is not None:
            video = video.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
        if torch.cuda.device_count() == 1 and args.gpu is None:
            video = video.cuda()
            target = target.cuda()

        # compute outputs
        if phase == 'test_dense':
            batch_size, clips_per_sample = video.shape[0], video.shape[1]
            video = video.flatten(0, 1).contiguous()

        if phase == 'train':
            logits = model(video)
        else:
            with torch.no_grad():
                logits = model(video)
        # print(logits.size())  # test torch.Size([32, 34])  # torch.Size([30, 34])
        # break
        if args.sample_grad_cam:
            logits[0, target[0].detach().cpu()].backward()
            gradients = model.module.get_activations_gradient()
            torch.save(gradients, "{}/video-heatmap-{:02d}-{:07d}-{:02d}.pt".format(args.data_address_heatmap, args.gpu,
                                                                                    it, target[0].detach().cpu()))
            # # print(gradients.size())
            # pooled_gradients = torch.mean(gradients[0, :, 0].unsqueeze(0), dim=[0, 2, 3])
            # # print(gradients[0, :, 0].unsqueeze(0).size())
            activations = model.module.get_activations(video).detach()
            torch.save(activations, "{}/activations-{:02d}-{:07d}.pt".format(args.data_address, args.gpu, it))
            # activations_handle = activations[:, :, 0, :, :]
            # for i in range(64):
            #     activations_handle[0, i, :, :] *= pooled_gradients[i]
            # heatmap = torch.mean(activations_handle[0].unsqueeze(0), dim=1).squeeze()
            # # heatmap = np.maximum(heatmap, 0)
            # # heatmap /= torch.max(heatmap)
            # # if args.gpu == 0:
            # torch.save(heatmap, "{}/video-heatmap-{:02d}-{:07d}.pt".format(args.data_address, args.gpu, it))
            # # plt.matshow(heatmap)
            # # plt.show()
            if phase == 'train':
                logits2 = model2(video)
            else:
                with torch.no_grad():
                    logits2 = model2(video)
            logits2[0, target[0].detach().cpu()].backward()
            gradients2 = model2.module.get_activations_gradient()
            torch.save(gradients2, "{}/video-heatmap-{:02d}-{:07d}-{:02d}.pt".format(args.data_address_heatmap_scratch, args.gpu,
                                                                                    it, target[0].detach().cpu()))
            activations2 = model2.module.get_activations(video).detach()
            torch.save(activations2, "{}/activations-{:02d}-{:07d}.pt".format(args.data_address_scratch, args.gpu, it))
        if args.sample_grad_cam:
            if (it + 1) % cfg['print_freq'] == 0 and args.gpu == 0:
                print('{:07d}/{:07d}'.format(it, len(loader)))
            if it >= 500:
                break
            continue
        # compute loss and accuracy
        if phase == 'test_dense':
            confidence = softmax(logits).view(batch_size, clips_per_sample, -1).mean(1)
            labels_tiled = target.unsqueeze(1).repeat(1, clips_per_sample).view(-1)
            loss = criterion(logits, labels_tiled)
        else:
            confidence = softmax(logits)
            loss = criterion(logits, target)

        with torch.no_grad():
            acc1, acc5 = metrics_utils.accuracy(confidence, target, topk=(1, 5))
            loss_meter.update(loss.item(), target.size(0))
            top1_meter.update(acc1[0], target.size(0))
            top5_meter.update(acc5[0], target.size(0))

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
            # print(index)

    if args.distributed and not args.sample_grad_cam:
        progress.synchronize_meters(args.gpu)
        progress.display(len(loader) * args.world_size, pr2term=False)
    
    if not args.sample_grad_cam:
        return top1_meter.avg, top5_meter.avg


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
