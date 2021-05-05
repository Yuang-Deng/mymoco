# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
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

        self.classifier = nn.Sequential()
        self.classifier.add_module('d_ln1', nn.Linear(2048, 10))
        self.classifier.add_module('d_softmax', nn.Softmax(dim=1))
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)
        # self.encoder_q_fc = self.encoder_q.fc
        # self.encoder_k_fc = self.encoder_k.fc
        # self.encoder_q.fc = nn.Sequential()
        # self.encoder_k.fc = nn.Sequential()

        # if mlp:  # hack: brute-force replacement
        #     dim_mlp = self.encoder_q.fc.weight.shape[1]
        #     self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
        #     self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # for param_q, param_k in zip(self.encoder_q_fc.parameters(), self.encoder_k_fc.parameters()):
        #     param_k.data.copy_(param_q.data)  # initialize
        #     param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        # for param_q, param_k in zip(self.encoder_q_fc.parameters(), self.encoder_k_fc.parameters()):
        #     param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

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

    def forward(self, im_labeled, im_q=None, im_k=None):
        if self.training:
            # compute query features
            # q, _ = modelforward(self.encoder_q, im_q)
            # q = self.encoder_q(im_q)  # queries: NxC
            # q = nn.functional.normalize(q, dim=1)

            q = modelforward(self.encoder_q, im_q)
            q_predict = self.classifier(q)
            q = self.encoder_q.fc(q)
            q = nn.functional.normalize(q, dim=1)

            feature = modelforward(self.encoder_q, im_labeled)  # queries: NxC

            classout = self.classifier(feature)

            with torch.no_grad():
                self._momentum_update_key_encoder()
                # k = self.encoder_k(im_k)  # keys: NxC
                # k = nn.functional.normalize(k, dim=1)

                k = modelforward(self.encoder_k, im_k)
                k_predict = self.classifier(k)
                k = self.encoder_k.fc(k)
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

            return logits, labels, classout, q_predict, k_predict
        else:
            labeled = modelforward(self.encoder_q, im_labeled)  # queries: NxC
            labeled = nn.functional.normalize(labeled, dim=1)
            classout = self.classifier(labeled)
            return classout


def modelforward(model, x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    x = x.view(x.shape[0], -1)
    return x
