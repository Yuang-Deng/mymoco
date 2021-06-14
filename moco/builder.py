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
        self.mocof = encoder_config['n_classes']
        self.encoder_qfc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(),
                                         nn.Linear(dim_mlp, encoder_config['n_classes']))
        self.encoder_kfc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(),
                                         nn.Linear(dim_mlp, encoder_config['n_classes']))

        self.classifier = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, 10))
        # self.classifier.add_module('d_ln1', nn.Linear(self.encoder_q.feature_size, 10))
        # self.classifier.add_module('d_softmax', nn.Softmax(dim=1))

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.encoder_qfc.parameters(), self.encoder_kfc.parameters()):
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
        for param_q, param_k in zip(self.encoder_qfc.parameters(), self.encoder_kfc.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)
        # 循环队列
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def contradict_forward(self, im_q=None, im_k=None, im_psupcon=None, p_label=None):
        q = self.encoder_q(im_q)
        q = self.encoder_qfc(q)
        q = nn.functional.normalize(q, dim=1)
        poslen = 0
        supdata = {}
        with torch.no_grad():
            for k, v in im_psupcon.items():
                poslen = len(v)
                supdata[k] = self.encoder_q(v)
                supdata[k] = self.encoder_qfc(supdata[k])
        supcon_data = torch.empty(size=[len(p_label), poslen, self.mocof]).cuda(0, non_blocking=True)
        for i in range(len(p_label)):
            supcon_data[i] = supdata[int(p_label[i])]

        with torch.no_grad():
            self._momentum_update_key_encoder()

            k = self.encoder_k(im_k)
            k = self.encoder_kfc(k)
            k = nn.functional.normalize(k, dim=1)

        supcon_data = torch.cat([k, supcon_data], dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k])
        l_pos = l_pos.unsqueeze(-1)
        l_ppos = torch.einsum('nc,nic->ni', [q, supcon_data])
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_ppos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        # pos_labels = torch.zeros([logits.shape[0], poslen + 1], dtype=torch.long).cuda()
        # neg_labels = torch.ones([logits.shape[0], len(l_neg[0])], dtype=torch.long).cuda()
        # labels = torch.cat([pos_labels, neg_labels], dim=1)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

    def contradict_forward_new(self, im_q=None, im_k=None, im_psupcon=None, p_label=None):
        q = self.encoder_q(im_q)
        q = self.encoder_qfc(q)
        q = nn.functional.normalize(q, dim=1)
        poslen = 0
        supdata = {}
        with torch.no_grad():
            for k, v in im_psupcon.items():
                poslen = len(v)
                supdata[k] = self.encoder_q(v)
                supdata[k] = self.encoder_qfc(supdata[k])
        supcon_data = torch.empty(size=[len(p_label), poslen, self.mocof]).cuda(0, non_blocking=True)
        for i in range(len(p_label)):
            supcon_data[i] = supdata[int(p_label[i])]

        with torch.no_grad():
            self._momentum_update_key_encoder()

            k = self.encoder_k(im_k)
            k = self.encoder_kfc(k)
            k = nn.functional.normalize(k, dim=1)
            k = k.view(128, -1, 128)

        pos_data = torch.cat([k, supcon_data], dim=1)
        l_pos = torch.einsum('nc,nic->ni', [q, pos_data])
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = []
        for i in range(poslen+1):
            l_p = l_pos[:, 1]
            logits.append(torch.cat([l_p, l_neg], dim=1))

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        # pos_labels = torch.zeros([logits.shape[0], poslen + 1], dtype=torch.long).cuda()
        # neg_labels = torch.ones([logits.shape[0], len(l_neg[0])], dtype=torch.long).cuda()
        # labels = torch.cat([pos_labels, neg_labels], dim=1)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

    # def forward(self, im_labeled, im_q=None, im_k=None, im_plabel=None):
    #     if self.training:
    #
    #         # 有标签损失
    #         im_labeled = self.encoder_q.forward(im_labeled)
    #         classout = self.classifier(im_labeled)
    #
    #         q = self.encoder_q.forward(im_q)
    #
    #         q_predict = self.classifier(q)
    #         q = self.encoder_q.fc(q)
    #         q = nn.functional.normalize(q, dim=1)
    #
    #         plabel = self.encoder_q.forward(im_plabel)
    #         p_predict = self.classifier(plabel)
    #
    #         with torch.no_grad():
    #             self._momentum_update_key_encoder()
    #
    #             k = self.encoder_k(im_k)
    #             k = nn.functional.normalize(k, dim=1)
    #
    #         l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
    #         # negative logits: NxK
    #         l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
    #
    #         # logits: Nx(1+K)
    #         logits = torch.cat([l_pos, l_neg], dim=1)
    #
    #         # apply temperature
    #         logits /= self.T
    #
    #         # labels: positive key indicators
    #         labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
    #
    #         # dequeue and enqueue
    #         self._dequeue_and_enqueue(k)
    #
    #         return logits, labels, classout, q_predict, p_predict
    #     else:
    #         labeled = self.encoder_q.forward_conv(im_labeled)  # queries: NxC
    #         labeled = labeled.view(labeled.size(0), -1)
    #         classout = self.classifier(labeled)
    #         return classout
