# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import nn
import torch.distributed as dist

import utils.logger
from utils import main_utils
import yaml
import os


def prepare_environment(args, cfg, fold):
    if args.distributed:
        while True:
            try:
                dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:{}'.format(args.port),
                                        world_size=args.world_size, rank=args.gpu)
                break
            except RuntimeError:
                args.port = str(int(args.port) + 1)

    model_cfg = yaml.safe_load(open(args.model_cfg))['model']
    eval_dir = '{}/{}/eval-{}/fold-{:02d}'.format(model_cfg['model_dir'],
                                                  model_cfg['name'], cfg['benchmark']['name'], fold)
    os.makedirs(eval_dir, exist_ok=True)
    num_file = 0
    if args.gpu == 0:
        while os.path.exists('{}/config{}.yaml'.format(eval_dir, num_file)):
            num_file += 1
        yaml.safe_dump(cfg, open('{}/config{}.yaml'.format(eval_dir, num_file), 'w'))
    num_file = 0
    while os.path.exists('{}/eval{}.log'.format(eval_dir, num_file)):
        num_file += 1
    logger = utils.logger.Logger(quiet=args.quiet, log_fn='{}/eval{}.log'.format(eval_dir, num_file), rank=args.gpu)
    if any(['SLURM' in env for env in list(os.environ.keys())]):
        logger.add_line("=" * 30 + "   SLURM   " + "=" * 30)
        for env in os.environ.keys():
            if 'SLURM' in env:
                logger.add_line('{:30}: {}'.format(env, os.environ[env]))
    logger.add_line("=" * 30 + "   Config   " + "=" * 30)

    def print_dict(d, ident=''):
        for k in d:
            if isinstance(d[k], dict):
                logger.add_line("{}{}".format(ident, k))
                print_dict(d[k], ident='  ' + ident)
            else:
                logger.add_line("{}{}: {}".format(ident, k, str(d[k])))

    print_dict(cfg)
    logger.add_line("=" * 30 + "   Model Config   " + "=" * 30)
    print_dict(model_cfg)

    return eval_dir, model_cfg, logger


def distribute_model_to_cuda(model, args, cfg):
    # if torch.cuda.device_count() == 1:
    #     model = model.cuda()
    # elif args.distributed:
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        if cfg['model']['name'] == 'MOSTWrapper-Audio':
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        model = torch.nn.DataParallel(model).cuda()
    return model


def build_dataloader(db_cfg, split_cfg, fold, num_workers, distributed):
    import torch.utils.data as data
    from datasets import preprocessing
    
    # else:
    #     raise ValueError
    return_video = True
    return_audio = False
    import datasets
    if db_cfg['name'] == 'ucf101':
        dataset = datasets.UCF
    elif db_cfg['name'] == 'hmdb51':
        dataset = datasets.HMDB
    elif db_cfg['name'] == 'kineticsdown':
        dataset = datasets.Kineticsdown
        return_video = True
        return_audio = True
    elif db_cfg['name'] == 'esc50':
        dataset = datasets.ESC
        return_video = False
        return_audio = True
    else:
        raise ValueError('Unknown dataset')

    if return_video and not return_audio:
        if db_cfg['transform'] == 'crop+color':
            video_transform = preprocessing.VideoPrep_Crop_CJ(
                crop=(db_cfg['crop_size'], db_cfg['crop_size']),
                num_frames=int(db_cfg['video_fps'] * db_cfg['clip_duration']),
                pad_missing=True,
                augment=split_cfg['use_augmentation'],
            )
        elif db_cfg['transform'] == 'msc+color':
            video_transform = preprocessing.VideoPrep_MSC_CJ(
                crop=(db_cfg['crop_size'], db_cfg['crop_size']),
                augment=split_cfg['use_augmentation'],
                num_frames=int(db_cfg['video_fps'] * db_cfg['clip_duration']),
                pad_missing=True,
                # min_area=db_cfg['min_area'],
                # color=db_cfg['color'],
            )
        else:
            raise ValueError
        db = dataset(
            subset=split_cfg['split'].format(fold=fold),
            return_video=return_video,
            video_clip_duration=db_cfg['clip_duration'],
            video_fps=db_cfg['video_fps'],
            video_transform=video_transform,
            return_audio=return_audio,
            return_labels=True,
            return_index=split_cfg['return_index'],
            mode=split_cfg['mode'],
            clips_per_video=split_cfg['clips_per_video'],
            sample_grad_cam=db_cfg['sample_grad_cam']
        )
    elif return_audio and not return_video:
        # Audio transforms
        audio_transforms = [
            preprocessing.AudioPrep(
                trim_pad=True,
                duration=db_cfg['clip_duration'],
                augment=split_cfg['use_augmentation'],
                missing_as_zero=True),
            preprocessing.LogSpectrogram(
                db_cfg['audio_fps'],
                n_fft=db_cfg['n_fft'],
                hop_size=1. / db_cfg['spectrogram_fps'],
                normalize=True)
        ]
        audio_fps_out = db_cfg['spectrogram_fps']
        db = dataset(
            subset=split_cfg['split'].format(fold=fold),
            return_video=return_video,
            return_audio=return_audio,
            audio_clip_duration=db_cfg['clip_duration'],
            audio_fps=db_cfg['audio_fps'],
            audio_fps_out=audio_fps_out,
            audio_transform=audio_transforms,
            return_labels=True,
            return_index=split_cfg['return_index'],
            mode=split_cfg['mode'],
            clips_per_video=split_cfg['clips_per_video'],
            max_offsync_augm=0.5 if split_cfg['use_augmentation'] else 0,
            sample_grad_cam_audio=db_cfg['sample_grad_cam_audio']
        )
    elif return_video and return_audio:
        if db_cfg['transform'] == 'crop+color':
            video_transform = preprocessing.VideoPrep_Crop_CJ(
                crop=(db_cfg['crop_size'], db_cfg['crop_size']),
                num_frames=int(db_cfg['video_fps'] * db_cfg['video_clip_duration']),
                pad_missing=True,
                augment=split_cfg['use_augmentation'],
            )
        elif db_cfg['transform'] == 'msc+color':
            video_transform = preprocessing.VideoPrep_MSC_CJ(
                crop=(db_cfg['crop_size'], db_cfg['crop_size']),
                augment=split_cfg['use_augmentation'],
                num_frames=int(db_cfg['video_fps'] * db_cfg['video_clip_duration']),
                pad_missing=True,
                # min_area=db_cfg['min_area'],
                # color=db_cfg['color'],
            )
        else:
            raise ValueError
        # Audio transforms
        audio_transforms = [
            preprocessing.AudioPrep(
                trim_pad=True,
                duration=db_cfg['audio_clip_duration'],
                augment=split_cfg['use_augmentation'],
                missing_as_zero=True),
            preprocessing.LogSpectrogram(
                db_cfg['audio_fps'],
                n_fft=db_cfg['n_fft'],
                hop_size=1. / db_cfg['spectrogram_fps'],
                normalize=True)
        ]
        audio_fps_out = db_cfg['spectrogram_fps']
        db = dataset(
            subset=split_cfg['split'].format(fold=fold),
            return_video=return_video,
            video_clip_duration=db_cfg['video_clip_duration'],
            video_fps=db_cfg['video_fps'],
            video_transform=video_transform,
            return_audio=return_audio,
            audio_clip_duration=db_cfg['audio_clip_duration'],
            audio_fps=db_cfg['audio_fps'],
            audio_fps_out=audio_fps_out,
            audio_transform=audio_transforms,
            return_labels=True,
            return_index=split_cfg['return_index'],
            mode='clip',
            clips_per_video=split_cfg['clips_per_video'],
            max_offsync_augm=0.5 if split_cfg['use_augmentation'] else 0,
            sample_grad_cam=db_cfg['sample_grad_cam'],
            sample_grad_cam_audio=db_cfg['sample_grad_cam_audio']
        )
    else:
        raise ValueError('Unknown return video or audio')

    use_shuffle = split_cfg['use_shuffle'] if 'use_shuffle' in split_cfg else True

    if distributed and use_shuffle:
        sampler = torch.utils.data.distributed.DistributedSampler(db)
    elif distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(db, shuffle=False)
    else:
        sampler = None

    drop_last = split_cfg['drop_last'] if 'drop_last' in split_cfg else True
    loader = data.DataLoader(
        db,
        batch_size=db_cfg['batch_size'] if split_cfg['mode'] == 'clip' else max(1, db_cfg['batch_size'] // split_cfg[
            'clips_per_video']),
        num_workers=num_workers,
        pin_memory=True,
        shuffle=(sampler is None) and use_shuffle,
        sampler=sampler,
        drop_last=drop_last
    )
    return loader


def build_dataloaders(cfg, fold, num_workers, distributed, logger):
    logger.add_line("=" * 30 + "   Train DB   " + "=" * 30)
    train_loader = build_dataloader(cfg, cfg['train'], fold, num_workers, distributed)
    logger.add_line(str(train_loader.dataset))

    logger.add_line("=" * 30 + "   Test DB   " + "=" * 30)
    test_loader = build_dataloader(cfg, cfg['test'], fold, num_workers, distributed)
    logger.add_line(str(test_loader.dataset))

    logger.add_line("=" * 30 + "   Dense DB   " + "=" * 30)
    dense_loader = build_dataloader(cfg, cfg['test_dense'], fold, num_workers, distributed)
    logger.add_line(str(dense_loader.dataset))

    return train_loader, test_loader, dense_loader


class CheckpointManager(object):
    def __init__(self, checkpoint_dir, rank=0):
        self.checkpoint_dir = checkpoint_dir
        self.best_metric = 0.
        self.rank = rank

    def save(self, model, optimizer, epoch, filename=None, eval_metric=0., num_file=0):
        if self.rank is not None and self.rank != 0:
            return
        is_best = False
        if eval_metric > self.best_metric:
            self.best_metric = eval_metric
            is_best = True
        if filename is None:
            main_utils.save_checkpoint(state={
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            }, is_best=is_best, model_dir=self.checkpoint_dir,num_file=num_file)
        else:
            main_utils.save_checkpoint(state={
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            }, is_best=is_best,model_dir=self.checkpoint_dir, filename='{}/{}'.format(self.checkpoint_dir, filename), num_file=num_file)


    def last_checkpoint_fn(self):
        return '{}/checkpoint.pth.tar'.format(self.checkpoint_dir)

    def best_checkpoint_fn(self):
        return '{}/model_best.pth.tar'.format(self.checkpoint_dir)

    def checkpoint_fn(self, last=False, best=False):
        assert best or last
        assert not (last and best)
        if last:
            return self.last_checkpoint_fn()
        if best:
            return self.best_checkpoint_fn()

    def checkpoint_exists(self, last=False, best=False):
        return os.path.isfile(self.checkpoint_fn(last, best))

    def restore(self, model, optimizer, restore_last=False, restore_best=False):
        checkpoint_fn = self.checkpoint_fn(restore_last, restore_best)
        ckp = torch.load(checkpoint_fn, map_location={'cuda:0': 'cpu'})
        start_epoch = ckp['epoch']
        model.load_state_dict(ckp['state_dict'])
        optimizer.load_state_dict(ckp['optimizer'])
        return start_epoch


class ClassificationWrapper(torch.nn.Module):
    def __init__(self, feature_extractor, n_classes, feat_name, feat_dim, pooling_op=None, use_dropout=False,
                 dropout=0.5):
        super(ClassificationWrapper, self).__init__()
        self.feature_extractor = feature_extractor
        self.feat_name = feat_name
        self.use_dropout = use_dropout
        if pooling_op is not None:
            self.pooling = eval('torch.nn.' + pooling_op)
        else:
            self.pooling = None
        if use_dropout:
            self.dropout = torch.nn.Dropout(dropout)
        self.classifier1 = torch.nn.Linear(feat_dim, 2048)
        self.classifier2 = torch.nn.Linear(2048, 1024)
        self.classifier3 = torch.nn.Linear(1024, 512)
        self.classifier4 = torch.nn.Linear(512, n_classes)

    def forward(self, *inputs):
        emb = self.feature_extractor(*inputs, return_embs=True)[self.feat_name]
        emb_pool = self.pooling(emb) if self.pooling is not None else emb
        emb_pool = emb_pool.view(inputs[0].shape[0], -1)
        if self.use_dropout:
            emb_pool = self.dropout(emb_pool)
        
        hid1 = self.classifier1(emb_pool)
        hid2 = self.classifier2(hid1)
        hid3 = self.classifier3(hid2)
        logit = self.classifier4(hid3)
        return logit


class ClassificationWrappercam(torch.nn.Module):
    def __init__(self, classification_net_eval):
        super(ClassificationWrappercam, self).__init__()
        self.classification_net_eval = classification_net_eval
        self.features_conx_visual = self.classification_net_eval.feature_extractor.conv1[0]
        self.bn_pool_visual = self.classification_net_eval.feature_extractor.conv1[1:]
        self.pool_visual = self.classification_net_eval.pooling
        self.dropout = self.classification_net_eval.dropout
        self.classifier = self.classification_net_eval.classifier
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, video):
        return self.features_conx_visual(video)

    # def extract_features(self, video, audio):
    #     video_emb = self.features_conx_visual(video)
    #     h = video_emb.register_hook(self.activations_hook)
    #     video_emb = self.bn_pool_visual(video_emb)
    #     video_emb = self.pool_visual(video_emb)
    #     audio_emb = self.audio_feat(audio)
    #     # video_emb = self.video_feat(video, return_embs=True)[self.feat_name_video]
    #     # audio_emb = self.audio_feat(audio, return_embs=True)[self.feat_name_audio]
    #     return video_emb, audio_emb

    # def classify(self, video_emb, audio_emb):
    #     emb = torch.cat((video_emb, audio_emb), 1)
    #     emb = self.dropout(emb)
    #     logit = self.classifier(emb)
    #     return logit

    def forward(self, video):
        emb = self.features_conx_visual(video)
        h = emb.register_hook(self.activations_hook)
        emb = self.bn_pool_visual(emb)
        emb_pool = self.pool_visual(emb)
        emb_pool = emb_pool.view(video.shape[0], -1)
        emb_pool = self.dropout(emb_pool)
        logit = self.classifier(emb_pool)
        return logit


class Classifier(nn.Module):
    def __init__(self, n_classes, feat_name, feat_dim, pooling, l2_norm=False, use_bn=True, use_dropout=False):
        super(Classifier, self).__init__()
        self.use_bn = use_bn
        self.feat_name = feat_name
        self.pooling = eval('nn.' + pooling) if pooling is not None else None
        self.l2_norm = l2_norm
        if use_bn:
            self.bn = nn.BatchNorm1d(feat_dim)
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout()
        self.classifier = nn.Linear(feat_dim, n_classes)

    def forward(self, x):
        with torch.no_grad():
            if self.use_dropout:
                x = self.dropout(x)
            if self.l2_norm:
                x = nn.functional.normalize(x, p=2, dim=-1)
            if self.pooling is not None and len(x.shape) > 2:
                x = self.pooling(x)
            x = x.view(x.shape[0], -1).contiguous().detach()
        if self.use_bn:
            x = self.bn(x)
        return self.classifier(x)


class MOSTCheckpointManager(object):
    def __init__(self, checkpoint_dir, rank=0):
        self.rank = rank
        self.checkpoint_dir = checkpoint_dir
        self.best_metric = 0.

    def save(self, model, optimizer, epoch, filename=None, eval_metric=0., num_file=0):
        if self.rank is not None and self.rank != 0:
            return
        is_best = False
        if eval_metric > self.best_metric:
            self.best_metric = eval_metric
            is_best = True
        # print(model)
        # print(model.state_dict())
        # try:
        #     state_dict = model.classifiers.state_dict()
        # except AttributeError:
        #     state_dict = model.module.classifiers.state_dict()
        if filename is None:
            main_utils.save_checkpoint(state={
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            }, is_best=is_best, model_dir=self.checkpoint_dir, num_file=num_file)
        else:
            main_utils.save_checkpoint(state={
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            }, is_best=is_best, model_dir=self.checkpoint_dir, filename='{}/{}'.format(self.checkpoint_dir, filename), num_file=num_file)
     

    def last_checkpoint_fn(self):
        return '{}/checkpoint.pth.tar'.format(self.checkpoint_dir)

    def best_checkpoint_fn(self):
        return '{}/model_best.pth.tar'.format(self.checkpoint_dir)

    def checkpoint_fn(self, last=False, best=False):
        assert best or last
        # assert not (last and best)
        if last:
            return self.last_checkpoint_fn()
        elif best:
            return self.best_checkpoint_fn()

    def checkpoint_exists(self, last=False, best=False):
        return os.path.isfile(self.checkpoint_fn(last, best))

    def restore(self, model, optimizer, restore_last=False, restore_best=False):
        checkpoint_fn = self.checkpoint_fn(restore_last, restore_best)
        ckp = torch.load(checkpoint_fn, map_location={'cuda:0': 'cpu'})
        start_epoch = ckp['epoch']
        model.load_state_dict(ckp['state_dict'])
        optimizer.load_state_dict(ckp['optimizer'])
        # try:
        #     model.classifiers.load_state_dict(ckp['state_dict'])
        # except AttributeError:
        #     model.module.classifiers.load_state_dict(ckp['state_dict'])
        # optimizer.load_state_dict(ckp['optimizer'])
        return start_epoch


class MOSTModel(nn.Module):
    def __init__(self, feature_extractor, n_classes, feat_names, feat_dims, pooling_ops, l2_norm=None, use_bn=False,
                 use_dropout=False, ft_all=False):
        super(MOSTModel, self).__init__()
        # assert len(feat_dims) == len(pooling_ops) == len(feat_names)
        # n_outputs = len(feat_dims)
        self.feat_names = feat_names
        self.feat_dims = feat_dims
        self.pooling_ops = pooling_ops
        if l2_norm is None:
            # l2_norm = [False] * len(feat_names)
            l2_norm = False
        # if not isinstance(l2_norm, list):
        #     l2_norm = [l2_norm] * len(feat_names)
        self.l2_norm = l2_norm
        self.ft_all = ft_all
        # feature_extractor.train(False)
        self.feature_extractor = feature_extractor

        # self.classifiers = nn.ModuleList([
        #     Classifier(n_classes, feat_name=feat_names[i], feat_dim=feat_dims[i], pooling=pooling_ops[i],
        #                l2_norm=l2_norm[i], use_bn=use_bn, use_dropout=use_dropout) for i in range(n_outputs)
        # ])
        self.classifiers = Classifier(n_classes, feat_name=feat_names, feat_dim=feat_dims, pooling=pooling_ops,
                       l2_norm=l2_norm, use_bn=use_bn, use_dropout=use_dropout)
        if not self.ft_all:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

    def forward(self, *x):
        if not self.ft_all:
            with torch.no_grad():
                embs = self.feature_extractor(*x, return_embs=self.feat_names)
                embs = embs[self.feat_names]
        else:
            embs = self.feature_extractor(*x, return_embs=self.feat_names)
            embs = embs[self.feat_names]
        # embs = self.feature_extractor(*x, return_embs=self.feat_names)
        # embs = embs[self.feat_names]
        logits = self.classifiers(embs)
        # for classifier, ft in zip(self.classifiers, self.feat_names):
        #     embs[ft] = classifier(embs[ft])
        return logits


class AVCWrapper(torch.nn.Module):
    def __init__(self, video_feat, audio_feat, feat_name_video, feat_name_audio, use_dropout=False, dropout=0.5):
        super(AVCWrapper, self).__init__()
        self.video_feat = video_feat
        self.audio_feat = audio_feat
        self.use_dropout = use_dropout
        self.feat_name_video = feat_name_video
        self.feat_name_audio = feat_name_audio
        if use_dropout:
            self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.video_feat.out_dim + self.audio_feat.out_dim, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 2),
        )

    def head_params(self):
        return self.classifier.parameters()

    def extract_features(self, video, audio):
        video_emb = self.video_feat(video, return_embs=True)[self.feat_name_video]
        audio_emb = self.audio_feat(audio, return_embs=True)[self.feat_name_audio]
        return video_emb, audio_emb

    def classify(self, video_emb, audio_emb):
        emb = torch.cat((video_emb, audio_emb), 1)
        if self.use_dropout:
            emb = self.dropout(emb)
        logit = self.classifier(emb)
        return logit

    def forward(self, video, audio):
        video_emb = self.video_feat(video, return_embs=True)[self.feat_name_video]
        audio_emb = self.audio_feat(audio, return_embs=True)[self.feat_name_audio]
        emb = torch.cat((video_emb, audio_emb), 1)
        if self.use_dropout:
            emb = self.dropout(emb)
        logit = self.classifier(emb)
        return logit


class AVCWrappercam(torch.nn.Module):
    def __init__(self, avc_net_eval):
        super(AVCWrappercam, self).__init__()
        self.avc_net_eval = avc_net_eval
        self.features_conx_visual = self.avc_net_eval.video_feat.conv1[0]
        self.bn_pool_visual = self.avc_net_eval.video_feat.conv1[1:]
        self.pool_visual = self.avc_net_eval.video_feat.pool
        self.audio_feat = self.avc_net_eval.audio_feat
        self.dropout = self.avc_net_eval.dropout
        self.classifier = self.avc_net_eval.classifier
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, video):
        return self.features_conx_visual(video)

    def extract_features(self, video, audio):
        video_emb = self.features_conx_visual(video)
        h = video_emb.register_hook(self.activations_hook)
        video_emb = self.bn_pool_visual(video_emb)
        video_emb = self.pool_visual(video_emb)
        audio_emb = self.audio_feat(audio)
        # video_emb = self.video_feat(video, return_embs=True)[self.feat_name_video]
        # audio_emb = self.audio_feat(audio, return_embs=True)[self.feat_name_audio]
        return video_emb, audio_emb

    def classify(self, video_emb, audio_emb):
        emb = torch.cat((video_emb, audio_emb), 1)
        emb = self.dropout(emb)
        logit = self.classifier(emb)
        return logit

    def forward(self, video, audio):
        video_emb = self.features_conx_visual(video)
        h = video_emb.register_hook(self.activations_hook)
        video_emb = self.bn_pool_visual(video_emb)
        video_emb = self.pool_visual(video_emb)
        audio_emb = self.audio_feat(audio)
        # video_emb = self.video_feat(video, return_embs=True)[self.feat_name_video]
        # audio_emb = self.audio_feat(audio, return_embs=True)[self.feat_name_audio]
        emb = torch.cat((video_emb, audio_emb), 1)
        emb = self.dropout(emb)
        logit = self.classifier(emb)
        return logit


def build_model(feat_cfg, eval_cfg, eval_dir, args, logger):
    import models
    import models_avc
    import models_adco
    if args.avc:
        pretrained_net = models_avc.__dict__[feat_cfg['arch']](**feat_cfg['args'])
    elif args.adco:
        pretrained_net = models_adco.__dict__[feat_cfg['arch']](**feat_cfg['args'])
    else:
        pretrained_net = models.__dict__[feat_cfg['arch']](**feat_cfg['args'])
        
    if not args.scratch:
        # Load from checkpoint
        checkpoint_fn = '{}/{}/checkpoint.pth.tar'.format(feat_cfg['model_dir'], feat_cfg['name'])
        ckp = torch.load(checkpoint_fn, map_location='cpu')
        if not args.avc:
            pretrained_net.load_state_dict({k.replace('module.', ''): ckp['base_model'][k] for k in ckp['base_model']})
        else:
            pretrained_net.load_state_dict({k.replace('module.', ''): ckp['model'][k] for k in ckp['model']})
    if not args.avc:
        # when avc is false, come in.
        # Wrap with linear-head classifiers
        if eval_cfg['model']['name'] == 'ClassificationWrapper':
            ckp_manager = CheckpointManager(eval_dir, rank=args.gpu)
            if not args.sample_grad_cam:
                model = ClassificationWrapper(feature_extractor=pretrained_net.video_model_q, **eval_cfg['model']['args'])
            else:
                pretrained_net_eval = ClassificationWrapper(feature_extractor=pretrained_net.video_model_q, **eval_cfg['model']['args'])
                checkpoint_fn_eval = '{}/checkpoint.pth.tar'.format(eval_dir)
                ckp_eval = torch.load(checkpoint_fn_eval, map_location='cpu')
                pretrained_net_eval.load_state_dict({k.replace('module.', ''): ckp_eval['state_dict'][k] for k in ckp_eval['state_dict']})
                model1 = ClassificationWrappercam(classification_net_eval=pretrained_net_eval)
                pretrained_net_eval2 = ClassificationWrapper(feature_extractor=pretrained_net.video_model_q, **eval_cfg['model']['args'])
                checkpoint_fn_eval2 = '{}/checkpoint.pth.tar'.format(eval_dir[:-8] + '_scratch' + eval_dir[-8:])
                ckp_eval2 = torch.load(checkpoint_fn_eval2, map_location='cpu')
                pretrained_net_eval2.load_state_dict(
                    {k.replace('module.', ''): ckp_eval2['state_dict'][k] for k in ckp_eval2['state_dict']})
                model2 = ClassificationWrappercam(classification_net_eval=pretrained_net_eval2)
                if args.distributed:
                    eval_cfg['dataset']['batch_size'] = max(eval_cfg['dataset']['batch_size'] // args.world_size, 1)
                    eval_cfg['num_workers'] = max(eval_cfg['num_workers'] // args.world_size, 1)
                model1 = distribute_model_to_cuda(model1, args, eval_cfg)
                model2 = distribute_model_to_cuda(model2, args, eval_cfg)
                return model1, model2, ckp_manager
        elif eval_cfg['model']['name'] == 'MOSTWrapper':
            model = MOSTModel(feature_extractor=pretrained_net.video_model_q, **eval_cfg['model']['args'])
            ckp_manager = MOSTCheckpointManager(eval_dir, rank=args.gpu)
        elif eval_cfg['model']['name'] == 'MOSTWrapper-Audio':
            model = MOSTModel(feature_extractor=pretrained_net.audio_model_q, **eval_cfg['model']['args'])
            ckp_manager = MOSTCheckpointManager(eval_dir, rank=args.gpu)
        elif eval_cfg['model']['name'] == 'avc_wrapper':
            ckp_manager = CheckpointManager(eval_dir, rank=args.gpu)
            if not args.sample_grad_cam:
                # when sample_grad_cam is false, come in.
                model = AVCWrapper(video_feat=pretrained_net.video_model_q, audio_feat=pretrained_net.audio_model_q, **eval_cfg['model']['args'])
            else:
                pretrained_net_eval = AVCWrapper(video_feat=pretrained_net.video_model_q, audio_feat=pretrained_net.audio_model_q, **eval_cfg['model']['args'])
                checkpoint_fn_eval = '{}/checkpoint.pth.tar'.format(eval_dir)
                ckp_eval = torch.load(checkpoint_fn_eval, map_location='cpu')
                pretrained_net_eval.load_state_dict({k.replace('module.', ''): ckp_eval['state_dict'][k] for k in ckp_eval['state_dict']})
                model = AVCWrappercam(avc_net_eval=pretrained_net_eval)
        else:
            raise ValueError
    else:
        # when avc is true, come in.
        # Wrap with linear-head classifiers
        if eval_cfg['model']['name'] == 'ClassificationWrapper':
            model = ClassificationWrapper(feature_extractor=pretrained_net.video_model, **eval_cfg['model']['args'])
            ckp_manager = CheckpointManager(eval_dir, rank=args.gpu)
        elif eval_cfg['model']['name'] == 'MOSTWrapper':
            model = MOSTModel(feature_extractor=pretrained_net.video_model, **eval_cfg['model']['args'])
            ckp_manager = MOSTCheckpointManager(eval_dir, rank=args.gpu)
        elif eval_cfg['model']['name'] == 'MOSTWrapper-Audio':
            model = MOSTModel(feature_extractor=pretrained_net.audio_model, **eval_cfg['model']['args'])
            ckp_manager = MOSTCheckpointManager(eval_dir, rank=args.gpu)
        elif eval_cfg['model']['name'] == 'avc_wrapper':
            model = AVCWrapper(video_feat=pretrained_net.video_model, audio_feat=pretrained_net.audio_model, **eval_cfg['model']['args'])
            ckp_manager = CheckpointManager(eval_dir, rank=args.gpu)
        else:
            raise ValueError

    # Log model description
    logger.add_line("=" * 30 + "   Model   " + "=" * 30)
    logger.add_line(str(model))
    logger.add_line("=" * 30 + "   Parameters   " + "=" * 30)
    logger.add_line(main_utils.parameter_description(model))
    logger.add_line("=" * 30 + "   Pretrained model   " + "=" * 30)
    # logger.add_line("File: {}\nEpoch: {}".format(checkpoint_fn, ckp['epoch']))

    # Distribute
    # if not args.sample_grad_cam:
    if args.distributed:
        eval_cfg['dataset']['batch_size'] = max(eval_cfg['dataset']['batch_size'] // args.world_size, 1)
        eval_cfg['num_workers'] = max(eval_cfg['num_workers'] // args.world_size, 1)
    model = distribute_model_to_cuda(model, args, eval_cfg)
    # else:
    #     model1 = distribute_model_to_cuda(model1, args, eval_cfg)
    #     model2 = distribute_model_to_cuda(model2, args, eval_cfg)
    #     model = [model1, model2]
    # logger.add_line(str(model))
    return model, ckp_manager


class BatchWrapper:
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

    def __call__(self, x):
        outs = []
        for i in range(0, x.shape[0], self.batch_size):
            outs += [self.model(x[i:i + self.batch_size])]  # torch.Size([64, 34])  # torch.Size([32, 34])
        return torch.cat(outs, 0)


def build_optimizer(params, cfg, logger=None):
    if cfg['name'] == 'sgd':
        optimizer = torch.optim.SGD(
            params=params,
            lr=cfg['lr']['base_lr'],
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay'],
            nesterov=cfg['nesterov']
        )

    elif cfg['name'] == 'adam':
        optimizer = torch.optim.Adam(
            params=params,
            lr=cfg['lr']['base_lr'],
            weight_decay=cfg['weight_decay'],
            betas=cfg['betas'] if 'betas' in cfg else [0.9, 0.999]
        )

    else:
        raise ValueError('Unknown optimizer.')

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['lr']['milestones'],
    #                                                  gamma=cfg['lr']['gamma'])
    return optimizer
