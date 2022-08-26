# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import nn

__all__ = ['AVAC']


class AdversarialSet(nn.Module):
    def __init__(self,
                 dim=128,
                 bank_size=1,
                 ):
        super(AdversarialSet, self).__init__()
        self.adversarial_samples = nn.Parameter(data=torch.randn([dim, bank_size]), requires_grad=True)
        
    def forward(self, q, update_memory=True):
        adversarial_samples = self.adversarial_samples
        if update_memory:
            adversarial_samples = nn.functional.normalize(adversarial_samples, dim=0)
            l_neg = torch.einsum('nc,ck->nk', q, adversarial_samples)
        else:
            with torch.no_grad():
                adversarial_samples = nn.functional.normalize(adversarial_samples, dim=0)
            l_neg = torch.einsum('nc,ck->nk', q, adversarial_samples.clone().detach())
        return l_neg


class AVAC(nn.Module):
    def __init__(self,
                 embedding_dim,
                 K=65536,
                 T_m=0.7,
                 T_n=0.07,
                 xModal_coeff=1.,
                 wModal_coeff=0.,
                 simple_model=True):
        super(AVAC, self).__init__()
        '''
        AVID criterion.
        This module receives the output embeddings of the video 
        and audio models, computes their non-linear projections, 
        manages the memory bank and computes the final loss.

        Args:
        - num_data: number of instances in the training set.
        - embedding_dim: output dimension of the non-linear projection.
        - num_negatives: number of negatives to draw from memory bank to compute the NCE loss.
        - momentum: memory bank EMA momemtum parameter.
        - xModal_coeff: coefficient for the cross modal loss. (Cross-AVID: 1.0 | Self-AVID: 0.0 | Joint-AVID: 1.0)
        - wModal_coeff: coefficient for the within modal loss. (Cross-AVID: 0.0 | Self-AVID: 1.0 | Joint-AVID: 1.0)
        - checkpoint: optinally specify a checkpoint path to restore the memory bank and partition function
        '''

        self.K = K
        self.T_m = T_m
        self.T_n = T_n
        if simple_model:
            self.Adversarial_av_set = AdversarialSet(embedding_dim, K)
        else:
            self.Adversarial_video_set = AdversarialSet(embedding_dim, K)
            self.Adversarial_audio_set = AdversarialSet(embedding_dim, K)
        sum_coeff = (xModal_coeff + wModal_coeff)
        self.xModal_coeff = xModal_coeff / sum_coeff
        self.wModal_coeff = wModal_coeff / sum_coeff
        self.criterion = nn.CrossEntropyLoss()
        self.simple_model = simple_model

    def forward(self, emb_v_q, emb_v_k, emb_a_q, emb_a_k, target, update_memory=False):
        '''
        Args
        - emb_v_q, emb_v_k: Video embeddings `(N, D)`
        - emb_a_q, emb_a_k: Audio embeddings `(N, D)`
        - taget: Intance labels `(N)`
        '''

        logits = {}

        if self.xModal_coeff > 0.:
            # positive logits: Nx1
            l_pos_va = torch.einsum('nc,nc->n', [emb_v_q, emb_a_k]).unsqueeze(-1)
            l_pos_av = torch.einsum('nc,nc->n', [emb_a_q, emb_v_k]).unsqueeze(-1)

            # negative logits: NxK
            if self.simple_model:
                l_neg_va_v = self.Adversarial_av_set(emb_v_q, update_memory=update_memory)
                l_neg_av_a = self.Adversarial_av_set(emb_a_q, update_memory=update_memory)
                # logits: Nx(1+K)
                logits['v2a'] = torch.cat([l_pos_va, l_neg_va_v], dim=1)
                logits['a2v'] = torch.cat([l_pos_av, l_neg_av_a], dim=1)
            else:
                l_neg_va = self.Adversarial_audio_set(emb_v_q, update_memory=update_memory)
                l_neg_av = self.Adversarial_video_set(emb_a_q, update_memory=update_memory)
                # logits: Nx(1+K)
                logits['v2a'] = torch.cat([l_pos_va, l_neg_va], dim=1)
                logits['a2v'] = torch.cat([l_pos_av, l_neg_av], dim=1)

        if self.wModal_coeff > 0.:
            # positive logits: Nx1
            l_pos_v = torch.einsum('nc,nc->n', [emb_v_q, emb_v_k]).unsqueeze(-1)
            l_pos_a = torch.einsum('nc,nc->n', [emb_a_q, emb_a_k]).unsqueeze(-1)

            # # negative logits: NxK
            if self.simple_model:
                # logits: Nx(1+K)
                logits['v2v'] = torch.cat([l_pos_v, l_neg_va_v], dim=1)
                logits['a2a'] = torch.cat([l_pos_a, l_neg_av_a], dim=1)
            else:
                l_neg_v = self.Adversarial_video_set(emb_v_q, update_memory=update_memory)
                l_neg_a = self.Adversarial_audio_set(emb_a_q, update_memory=update_memory)
                # logits: Nx(1+K)
                logits['v2v'] = torch.cat([l_pos_v, l_neg_v], dim=1)
                logits['a2a'] = torch.cat([l_pos_a, l_neg_a], dim=1)


        xModal_loss, wModal_loss = 0., 0.
        for k in logits:
            logit = logits[k]
            if update_memory:
                logit /= self.T_n
            else:
                logit /= self.T_m
            labels = torch.zeros(logit.shape[0], dtype=torch.long).cuda()
            loss = self.criterion(logit, labels)
            if k in {'v2a', 'a2v'}:
                xModal_loss += loss / 2.
            elif k in {'v2v', 'a2a'}:
                wModal_loss += loss / 2.

        if update_memory:
            total_loss = - xModal_loss * self.xModal_coeff - wModal_loss * self.wModal_coeff
        else:
            total_loss = xModal_loss * self.xModal_coeff + wModal_loss * self.wModal_coeff

        return total_loss

    def set_epoch(self, epoch):
        pass
