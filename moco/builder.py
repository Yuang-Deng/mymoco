# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, encoder_config, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = base_encoder(encoder_config)
        self.encoder_k = base_encoder(encoder_config)

        dim_mlp = self.encoder_q.feature_size
        self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
        self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        self.classifier = nn.Sequential()
        self.classifier.add_module('d_ln1', nn.Linear(self.encoder_q.feature_size, 10))
        self.classifier.add_module('d_softmax', nn.Softmax(dim=1))

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(encoder_config['n_classes'], K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_labeled, im_q=None, im_k=None, im_plabel=None):
        if self.training:
            # imgs = torch.cat([im_labeled, im_q, im_plabel], dim=0)
            imgs = torch.cat([im_labeled, im_q, im_plabel], dim=0)
            imgs = self.encoder_q.forward_conv(imgs)
            imgs = imgs.view(imgs.size(0), -1)
            feature = imgs[:im_labeled.size(0)]
            q, plabel = torch.split(imgs[im_labeled.size(0):], im_q.size(0))

            # q = self.encoder_q.forward_conv(im_q)
            # q = q.view(q.size(0), -1)
            q_predict = self.classifier(q)
            q = self.encoder_q.fc(q)
            q = nn.functional.normalize(q, dim=1)

            # plabel = self.encoder_q.forward_conv(im_plabel)
            # plabel = plabel.view(plabel.size(0), -1)
            p_predict = self.classifier(plabel)

            # feature = self.encoder_q.forward_conv(im_labeled)  # queries: NxC
            # feature = feature.view(feature.size(0), -1)
            classout = self.classifier(feature)

            with torch.no_grad():
                self._momentum_update_key_encoder()

                k = self.encoder_k(im_k)
                k = nn.functional.normalize(k, dim=1)

            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.T

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

            # dequeue and enqueue
            self._dequeue_and_enqueue(k)

            return logits, labels, classout, q_predict, p_predict
        else:
            labeled = self.encoder_q.forward_conv(im_labeled)  # queries: NxC
            labeled = labeled.view(labeled.size(0), -1)
            classout = self.classifier(labeled)
            return classout
