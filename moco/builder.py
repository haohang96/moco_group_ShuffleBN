# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import torch
import torch.nn as nn

from .head import Head

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, head_type, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
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
        self.s3_size = 8 # width(height) of selected patch in stage 3

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        self.encoder_q.add_head = Head(head_type, 1024)
        self.encoder_k.add_head = Head(head_type, 1024)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer('s3_queue', torch.randn(dim, K))
        self.s3_queue = nn.functional.normalize(self.s3_queue, dim=0)


        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, s3_keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        s3_keys = concat_all_gather(s3_keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.s3_queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _group_batch_shuffle_ddp(self, x, gpu_rank, node_rank, ngpu_per_node, nrank_per_subg, groups):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather, select_subg = group_concat_all_gather(x, gpu_rank, node_rank, ngpu_per_node, nrank_per_subg, groups)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this # here means num_gpus per subg

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        # src rank in select_subg
        src_rank = (gpu_rank//nrank_per_subg)*nrank_per_subg + node_rank*ngpu_per_node
        torch.distributed.broadcast(idx_shuffle, src=src_rank, group=select_subg)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = gpu_rank%nrank_per_subg # gpu_rank in each select_subg
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _group_batch_unshuffle_ddp(self, x, idx_unshuffle, gpu_rank, node_rank, ngpu_per_node, nrank_per_subg, groups):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather,_ = group_concat_all_gather(x, gpu_rank, node_rank, ngpu_per_node, nrank_per_subg, groups)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = gpu_rank%nrank_per_subg # gpu_rank in each select_subg
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]


    def gen_randbbox(self, feat_size, out_size):
        w_box = out_size
        h_box = w_box
        start_rid = np.random.randint(feat_size - w_box)
        start_cid = np.random.randint(feat_size - h_box)
        end_rid = start_rid + h_box
        end_cid = start_cid + w_box
        return (start_rid, start_cid, end_rid, end_cid)


    @torch.no_grad()
    def find_q_coor(self, q_feat, k_feat, stride=3):
        start_rid = 0
        end_rid = q_feat.size(2) - k_feat.size(2)
        start_cid = 0
        end_cid = q_feat.size(3) - k_feat.size(3)
        w = k_feat.size(-1)
        bs = k_feat.size(0)

        # how many times can sliding k-patch on q feat in one direction
        num_on_row = (end_rid // stride) + 1
        num_on_col = (end_cid // stride) + 1

        sim = torch.zeros(bs,num_on_row**2)

        q_feat = nn.functional.normalize(q_feat, dim=1)
        k_feat = nn.functional.normalize(k_feat, dim=1)

        for i in range(start_rid, end_rid+1, stride):
            for j in range(start_cid, end_cid+1, stride):
                q_patch = q_feat[:,:,i:i+w,j:j+w]
                _sim = (q_patch*k_feat).sum(-1).sum(-1).sum(-1)
                sim[:,(i//stride)*num_on_row + j//stride] = _sim

        max_sim_idx = sim.max(1)[1]

        # start rid and cid of selected q patch
        selected_q_rid = (max_sim_idx // num_on_row) * stride
        selected_q_cid = (max_sim_idx %  num_on_col) * stride


        return selected_q_rid, selected_q_cid


    def forward(self, im_q, im_k, gpu_rank, node_rank, ngpu_per_node, nrank_per_subg, groups):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            gpu_rank: gpu rank in one node(0 to FLAGS.ngpu)
            node_rank: node rank(0 to FLAGS.nodes_num)
            groups: a list of subgroup, enable shuffle bn in each sub-group
        Output:
            logits, targets
        """

        # compute query features
        mat_s3_q, q = self.encoder_q(im_q, local_out=True)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._group_batch_shuffle_ddp(im_k, gpu_rank, node_rank, ngpu_per_node, nrank_per_subg, groups)

            mat_s3_k, k = self.encoder_k(im_k, local_out=True)  # keys: NxC

            s3_r1, s3_c1, s3_r2, s3_c2 = self.gen_randbbox(mat_s3_k.size(-1), self.s3_size)
            mat_s3_k = mat_s3_k[:,:, s3_r1:s3_r2, s3_c1:s3_c2].clone()

            k = nn.functional.normalize(k, dim=1)
            s3_k = self.encoder_k.add_head(mat_s3_k)
            s3_k = nn.functional.normalize(s3_k, dim=1)


            # undo shuffle
            k = self._group_batch_unshuffle_ddp(k, idx_unshuffle, gpu_rank, node_rank, ngpu_per_node, nrank_per_subg, groups)
            s3_k = self._group_batch_unshuffle_ddp(s3_k, idx_unshuffle, gpu_rank, node_rank, ngpu_per_node, nrank_per_subg, groups)
            mat_s3_k = self._group_batch_unshuffle_ddp(mat_s3_k, idx_unshuffle, gpu_rank, node_rank, ngpu_per_node, nrank_per_subg, groups)



        with torch.no_grad():
            s3_q_k_coor = self.find_q_coor(mat_s3_q, mat_s3_k)

        s3_q = self.gen_patch_by_coor(mat_s3_q, s3_q_k_coor, self.s3_size)
        s3_q = self.encoder_q.add_head(s3_q)
        s3_q = nn.functional.normalize(s3_q, dim=1)


        l_s3_pos = torch.einsum('nc,nc->n', [s3_q, s3_k]).unsqueeze(-1)
        l_s3_neg = torch.einsum('nc,ck->nk', [s3_q, self.s3_queue.clone().detach()])
        logits_s3 = torch.cat([l_s3_pos, l_s3_neg], dim=1)
        logits_s3 /= self.T

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
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
        self._dequeue_and_enqueue(k, s3_k)

        return logits, logits_s3, labels


    def gen_patch_by_coor(self, q, coor, w):
        _bs = q.size(0)
        channel = q.size(1)

        #rid, cid = coor
        rid = coor[0]
        cid = coor[1]
        rid = rid.reshape(-1,1).repeat(1,w)
        cid = cid.reshape(-1,1).repeat(1,w)

        offset = torch.arange(w)
        rid = (rid + offset).reshape(-1,1).repeat(1,w).reshape(_bs,-1).reshape(-1)
        cid = (cid + offset).repeat(1,w).reshape(-1)

        batch_id = torch.arange(_bs).reshape(-1,1).repeat(1,w*w).reshape(-1)


        q = q[batch_id, :, rid, cid]
        q = q.reshape(_bs, -1, channel)
        q = q.reshape(_bs, w, w, channel)
        q = q.permute(0,3,1,2)

        return q



# utils
@torch.no_grad()
def group_concat_all_gather(tensor, gpu_rank, node_rank, ngpu_per_node, nrank_per_subg, groups):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    assert ngpu_per_node//nrank_per_subg == len(groups)
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(nrank_per_subg)]
    select_subg_idx = gpu_rank // nrank_per_subg
    select_subg = groups[select_subg_idx]
    torch.distributed.all_gather(tensors_gather, tensor, select_subg, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output, select_subg

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

