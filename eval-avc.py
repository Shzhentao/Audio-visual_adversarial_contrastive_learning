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
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Evaluation on Audio-Visual Correspondance')
parser.add_argument('cfg', metavar='CFG', help='eval config file')
parser.add_argument('model_cfg', metavar='MODEL_CFG', help='model config file')
parser.add_argument('--quiet', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--test-only', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--port', default='1234')
parser.add_argument('--crop-acc', action='store_true')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
parser.add_argument('--scratch', action='store_true', help='scratch training')
parser.add_argument('--avc', action='store_true', help='scratch training')
parser.add_argument('--sample_grad_cam', action='store_true', help='sample grad cam')


def main():
    ngpus = torch.cuda.device_count()
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    # if 'scratch' not in cfg:
    #     cfg['scratch'] = False
    # if 'ft_all' not in cfg:
    #     cfg['ft_all'] = False
    if args.test_only:
        cfg['test_only'] = True
    if args.resume:
        cfg['resume'] = True
    if args.debug:
        cfg['num_workers'] = 1
        cfg['dataset']['batch_size'] = 4
    if args.sample_grad_cam:
        cfg['dataset']['sample_grad_cam'] = True
        cfg['dataset']['sample_grad_cam_audio'] = True
        cfg['num_workers'] = 1
        cfg['dataset']['batch_size'] = 1
        data_addresses = './data/kinetics/{}'
        data_list = ['visual', 'audio']
        for index in range(2):
            data_address = data_addresses.format(data_list[index])
            if not os.path.exists(data_address):
                os.makedirs(data_address)
            else:
                shutil.rmtree(data_address)
                os.makedirs(data_address)
        args.data_address_visual = data_addresses.format(data_list[0])
        args.data_address_audio = data_addresses.format(data_list[1])

    if args.distributed:
        mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, cfg['dataset']['fold'], args, cfg))
    else:
        main_worker(0, ngpus, cfg['dataset']['fold'], args, cfg)


def main_worker(gpu, ngpus, fold, args, cfg):
    args.gpu = gpu
    args.world_size = ngpus

    # Prepare for training
    eval_dir, model_cfg, logger = eval_utils.prepare_environment(args, cfg, fold)

    # Model
    model, ckp_manager = eval_utils.build_model(model_cfg, cfg, eval_dir, args, logger)

    # Optimizer
    optimizer = eval_utils.build_optimizer(model.parameters(), cfg['optimizer'], logger)
    
    # Datasets
    train_loader, test_loader, _ = eval_utils.build_dataloaders(cfg['dataset'], fold, cfg['num_workers'], args.distributed, logger)

    # Optionally resume from a checkpoint
    start_epoch, end_epoch = 0, cfg['optimizer']['num_epochs']
    if (cfg['resume'] or cfg['test_only']) and not args.sample_grad_cam:
        if ckp_manager.checkpoint_exists(last=True):
                start_epoch = ckp_manager.restore(model, optimizer, restore_last=True)
        else:
            raise AttributeError

    # ######################## TRAINING #########################
    
    if not cfg['test_only']:
        logger.add_line("=" * 30 + "   Training   " + "=" * 30)
        gamma = cfg['optimizer']['lr']['gamma']
        num_file = 0
        while os.path.exists('{}/model_best{}.pth.tar'.format(eval_dir, num_file)):
            num_file += 1
        for epoch in range(start_epoch, end_epoch):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                test_loader.sampler.set_epoch(epoch)
            adjust_learning_rate(optimizer, cfg['optimizer']['lr']['base_lr'],
                                 epoch, end_epoch, gamma, cfg['optimizer']['lr']['milestones'], args)
            # train_loader.dataset.shuffle_dataset()
            # test_loader.dataset.shuffle_dataset()
            logger.add_line('='*30 + ' Epoch {} '.format(epoch) + '='*30)
            logger.add_line('LR: {}'.format(get_learning_rate(optimizer)))
            if args.sample_grad_cam:
                run_phase('train', train_loader, model, optimizer, epoch, args, cfg, logger)
                break
            run_phase('train', train_loader, model, optimizer, epoch, args, cfg, logger)
            top1 = run_phase('test', test_loader, model, None, epoch, args, cfg, logger)
            ckp_manager.save(model, optimizer, epoch, eval_metric=top1,num_file=num_file)
            save_freq = cfg['optimizer']['lr']['save_freq']
            if (epoch+1) % save_freq == 0:
                ckp_manager.save(model, optimizer, epoch, filename='checkpoint-ep{}.pth.tar'.format(epoch),num_file=num_file)
    
    if not args.sample_grad_cam:
        # ######################## TESTING #########################
        logger.add_line('\n' + '=' * 30 + ' Final evaluation ' + '=' * 30)
        top1 = run_phase('test', test_loader, model, None, end_epoch, args, cfg, logger)

        # ######################## LOG RESULTS #########################
        logger.add_line('\n' + '=' * 30 + ' Evaluation done ' + '=' * 30)
        logger.add_line('Clip@1: {:6.2f}'.format(top1))


def run_phase(phase, loader, model, optimizer, epoch, args, cfg, logger):
    from utils import metrics_utils
    batch_time = metrics_utils.AverageMeter('Time', ':6.6f', window_size=100)
    data_time = metrics_utils.AverageMeter('Data', ':6.6f', window_size=100)
    loss_meter = metrics_utils.AverageMeter('Loss', ':.4e')
    acc_meter = metrics_utils.AverageMeter('Acc@1', ':6.6f')
    progress = utils.logger.ProgressMeter(len(loader), meters=[batch_time, data_time, loss_meter, acc_meter],
                                        phase=phase, epoch=epoch, logger=logger)

    # try:
    #     audio_channels = model.audio_feat.conv1[0].in_channels
    #     model.extract_features
    # except AttributeError:
    #     audio_channels = model.module.audio_feat.conv1[0].in_channels
    #     model.extract_features = model.module.extract_features
    #     model.classify = model.module.classify

    # switch to train/test mode
    model.train(phase == 'train')
    # model.classifier.train(phase == 'train')
    # if align_criterion is not None:
    #     align_criterion.train(phase == 'train' and cfg['ft_all'])

    end = time.time()
    logger.add_line('\n{}: Epoch {}'.format(phase, epoch))
    criterion = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)
    # print(len(loader))
    for it, sample in enumerate(loader):
        if not args.sample_grad_cam:
            data_time.update(time.time() - end)

        # bs, n_aug = sample['video'].shape[:2]

        # video = sample['video'].flatten(0, 1)
        # audio = sample['audio'].flatten(0, 1)
        video = sample['frames0']
        # print(video.size())
        audio = sample['audio0']
        # print(audio.size())
        # target = sample['label'].cuda()
        if args.sample_grad_cam and not phase == 'test_dense':
            video_origin = sample['frames2']
            # print(video_origin.size())
            torch.save(video_origin, "{}/video-origin-{:02d}-{:07d}.pt".format(args.data_address_visual, args.gpu, it))
            torch.save(video, "{}/video{:02d}-{:07d}.pt".format(args.data_address_visual, args.gpu, it))
            audio_origin = sample['audio2']
            # print(audio_origin.size())
            torch.save(audio_origin, "{}/audio-origin-{:02d}-{:07d}.pt".format(args.data_address_audio, args.gpu, it))
            torch.save(audio, "{}/audio{:02d}-{:07d}.pt".format(args.data_address_audio, args.gpu, it))
        if args.gpu is not None:
            video = video.cuda(args.gpu, non_blocking=True)
            audio = audio.cuda(args.gpu, non_blocking=True)
        if torch.cuda.device_count() == 1 and args.gpu is None:
            video = video.cuda()
            audio = audio.cuda()
        # # compute outputs
        with torch.set_grad_enabled(phase == 'train'):
            video_emb, audio_emb = model.module.extract_features(video, audio)
            if not args.sample_grad_cam:
                video_emb, audio_emb, labels = create_avc_pairs(video_emb, audio_emb, args)
            # print(labels.size())
            # video_emb = video_emb.flatten(0, 1)
            # audio_emb = audio_emb.flatten(0, 1)
        # print(video_emb.size())  # torch.Size([64, 64, 1, 1, 1])
        # print(audio_emb.size())  # torch.Size([64, 64, 1, 1])
        video_emb = video_emb.flatten(start_dim=1)
        audio_emb = audio_emb.flatten(start_dim=1)
        with torch.set_grad_enabled(phase == 'train'):
            logits = model.module.classify(video_emb, audio_emb)
        # print(logits.size())
        # print(labels.size())
        # print(labels.dtype)
        if args.sample_grad_cam:
            logits[0, 0].backward()
            gradients = model.module.get_activations_gradient()
            # print(gradients.size())
            pooled_gradients = torch.mean(gradients[0, :, 0].unsqueeze(0), dim=[0, 2, 3])
            # print(gradients[0, :, 0].unsqueeze(0).size())
            activations = model.module.get_activations(video).detach()
            activations_handle = activations[:, :, 0, :, :]
            for i in range(64):
                activations_handle[0, i, :, :] *= pooled_gradients[i]
            heatmap = torch.mean(activations_handle[0].unsqueeze(0), dim=1).squeeze()
            # heatmap = np.maximum(heatmap, 0)
            # heatmap /= torch.max(heatmap)
            # if args.gpu == 0:
            torch.save(heatmap, "{}/video-heatmap-{:02d}-{:07d}.pt".format(args.data_address_visual, args.gpu, it))
            # plt.matshow(heatmap)
            # plt.show()
        if args.sample_grad_cam:
            if (it + 1) % cfg['print_freq'] == 0 and args.gpu == 0:
                print('{:07d}/{:07d}'.format(it, len(loader)))
            continue
        # compute loss and measure accuracy
        confidence = softmax(logits)
        loss = criterion(logits, labels)

        with torch.no_grad():
            acc1 = metrics_utils.accuracy(confidence, labels, topk=(1, ))[0]
            loss_meter.update(loss.item(), labels.size(0))
            acc_meter.update(acc1[0], labels.size(0))

        # compute gradient and do SGD step
        if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (it + 1) % cfg['print_freq'] == 0 or it == 0 or it + 1 == len(loader):
            progress.display(it+1, pr2term=False)

    if args.distributed and not args.sample_grad_cam:
        progress.synchronize_meters(args.gpu)
        progress.display(len(loader) * args.world_size, pr2term=False)

    if not args.sample_grad_cam:
        return acc_meter.avg


def create_avc_pairs(video_embs, audio_embs, args):
    bs = video_embs.shape[0]
    rnd_idx = torch.randint(0, bs - 1, (bs,))
    rnd_idx = rnd_idx + (rnd_idx >= torch.arange(0, bs)).int()
    video_embs = torch.cat((video_embs, video_embs), 0)
    audio_embs = torch.cat((audio_embs, audio_embs[rnd_idx]), 0)
    labels = torch.cat((torch.zeros(bs, ), torch.ones(bs, )), 0).long()
    if args.gpu is not None:
        labels = labels.cuda(args.gpu, non_blocking=True)
    return video_embs, audio_embs, labels


# def build_dataloaders(cfg, num_workers, distributed, logger):
#     logger.add_line("=" * 30 + "   Train DB   " + "=" * 30)
#     train_loader = main_utils.build_dataloader(cfg, cfg['train'], num_workers, distributed)
#     logger.add_line(str(train_loader.dataset))

#     logger.add_line("=" * 30 + "   Test DB   " + "=" * 30)
#     test_loader = main_utils.build_dataloader(cfg, cfg['test'], num_workers, distributed)
#     logger.add_line(str(test_loader.dataset))

#     return train_loader, test_loader


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
