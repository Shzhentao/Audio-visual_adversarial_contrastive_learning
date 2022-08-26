# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from random import shuffle
import torch
import torch.nn as nn


__all__ = [
    'av_wrapper'
]


class Head(nn.Module):
    def __init__(self, input_dim, proj_dims):
        super(Head, self).__init__()
        if not isinstance(proj_dims, list):
            proj_dims = [proj_dims]

        projection = []
        for i, d in enumerate(proj_dims):
            projection += [nn.Linear(input_dim, d)]
            input_dim = d
            if i < len(proj_dims)-1:
                projection += [nn.ReLU(inplace=True)]
        self.projection = nn.Sequential(*projection)
        self.out_dim = proj_dims[-1]

    def forward(self, x):
        return self.projection(x)


class AV_Wrapper(nn.Module):
    def __init__(self, video_models, proj_dim=128, use_shuffle=False):
        super(AV_Wrapper, self).__init__()
        self.use_shuffle = use_shuffle
        self.video_model_q = video_models[0]
        self.video_model_k = video_models[1]
        
        for param_q, param_k in zip(self.video_model_q.parameters(), self.video_model_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.use_linear_proj = proj_dim is not None
        if proj_dim is not None:
            self.video_proj_q = Head(video_models[0].out_dim, proj_dim)
            self.video_proj_k = Head(video_models[0].out_dim, proj_dim)

            for param_q, param_k in zip(self.video_proj_q.parameters(), self.video_proj_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

            self.out_dim = self.video_proj_q.out_dim
            
        else:
            self.out_dim = self.video_model_q.out_dim

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle
    
    @torch.no_grad()
    def _update_key_encoder(self):
        """
        Update of the key encoder
        """
        for param_q, param_k in zip(self.video_model_q.parameters(), self.video_model_k.parameters()):
            param_k.data.copy_(param_q.data)
            
        for param_q, param_k in zip(self.audio_model_q.parameters(), self.audio_model_k.parameters()):
            param_k.data.copy_(param_q.data)

    def forward(self, video_q, video_k, audio_q, audio_k):
        video_emb_q = self.video_model_q(video_q)
        video_emb_q = video_emb_q.view(video_emb_q.shape[0], video_emb_q.shape[1])
        if self.use_linear_proj:
            video_emb_q = self.video_proj_q(video_emb_q)

        audio_emb_q = self.audio_model_q(audio_q)
        audio_emb_q = audio_emb_q.view(audio_emb_q.shape[0], audio_emb_q.shape[1])
        if self.use_linear_proj:
            audio_emb_q = self.audio_proj_q(audio_emb_q)
            
        with torch.no_grad():
            self._update_key_encoder()  # update the key encoder
            if self.use_shuffle:
                # shuffle for making use of BN
                video_k, idx_unshuffle_v = self._batch_shuffle_ddp(video_k)
            video_emb_k = self.video_model_k(video_k)
            video_emb_k = video_emb_k.view(video_emb_k.shape[0], video_emb_k.shape[1])  # keys: NxC
            if self.use_linear_proj:
                video_emb_k = self.video_proj_k(video_emb_k)

            if self.use_shuffle:
                # shuffle for making use of BN
                audio_k, idx_unshuffle_a = self._batch_shuffle_ddp(audio_k)
            audio_emb_k = self.audio_model_k(audio_k)
            # audio_emb_k = self.audio_model_k(audio_k)
            audio_emb_k = audio_emb_k.view(audio_emb_k.shape[0], audio_emb_k.shape[1])  # keys: NxC
            if self.use_linear_proj:
                audio_emb_k = self.audio_proj_k(audio_emb_k)
        
        video_emb_q = nn.functional.normalize(video_emb_q, dim=1)
        audio_emb_q = nn.functional.normalize(audio_emb_q, dim=1)
        with torch.no_grad():
            video_emb_k = nn.functional.normalize(video_emb_k, dim=1)
            audio_emb_k = nn.functional.normalize(audio_emb_k, dim=1)
            if self.use_shuffle:
                # undo shuffle
                video_emb_k = self._batch_unshuffle_ddp(video_emb_k, idx_unshuffle_v)
                audio_emb_k = self._batch_unshuffle_ddp(audio_emb_k, idx_unshuffle_a)
        
        return video_emb_q, video_emb_k, audio_emb_q, audio_emb_k


def av_wrapper(video_backbone, video_backbone_args, audio_backbone, audio_backbone_args, proj_dim=128, checkpoint=None, use_shuffle=False):
    import models
    assert video_backbone in models.__dict__, 'Unknown model architecture'
    assert audio_backbone in models.__dict__, 'Unknown model architecture'
    video_model_q = models.__dict__[video_backbone](**video_backbone_args)
    video_model_k = models.__dict__[video_backbone](**video_backbone_args)
    video_models = [video_model_q, video_model_k]
    model = AV_Wrapper(video_models, proj_dim=proj_dim, use_shuffle=use_shuffle)
    if checkpoint is not None:
        ckp = torch.load(checkpoint, map_location='cpu')
        nn.DataParallel(model).load_state_dict(ckp['model'])

    return model


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
