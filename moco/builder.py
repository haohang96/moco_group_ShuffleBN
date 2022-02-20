# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import pdb
import torch
import torch.nn as nn

from timm.models import create_model
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from einops import rearrange
from torchvision.transforms import ToPILImage
from moco import modeling_pretrain


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, mae_aug_prob=0.):
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
        self.mae_aug_prob = mae_aug_prob

        self.mae_model = create_model(
            'pretrain_mae_base_patch16_224',
            pretrained=False,
            drop_path_rate=0.,
            drop_block_rate=None)
        pretrained_mae = torch.load('pretrain_mae_vit_base_mask_0.75_400e.pth')
        self.mae_model.load_state_dict(pretrained_mae['model'])

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

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
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
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

    def forward(self, im_q, im_k, batch_mae, gpu_rank, node_rank, ngpu_per_node, nrank_per_subg, groups):
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
        _bs = im_q.size(0)
        x_mae, bool_masked_pos = batch_mae
        x_mae, bool_masked_pos = x_mae.cuda(), bool_masked_pos.cuda().flatten(1).to(torch.bool)
        if torch.rand(1) < 1.1:
            with torch.no_grad():
                patch_size = 16
                mae_aug_num = int(_bs * self.mae_aug_prob)
                mae_aug_idx = torch.randperm(_bs)[:mae_aug_num]

                x_mae = x_mae[mae_aug_idx]
                bool_masked_pos = bool_masked_pos[mae_aug_idx]

                outputs = self.mae_model(x_mae, bool_masked_pos)

                #save original img
                mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).cuda()[None, :, None, None]
                std = torch.as_tensor(IMAGENET_DEFAULT_STD).cuda()[None, :, None, None]
                ori_img = x_mae * std + mean  # in [0, 1]

                img_squeeze = rearrange(ori_img, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size, p2=patch_size)
                img_norm = (img_squeeze - img_squeeze.mean(dim=-2, keepdim=True)) / (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')

                for i in range(img_patch.size(0)):
                    img_patch[i,bool_masked_pos[i]] = outputs[i].to(torch.float)


                rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=3)
                rec_img = rec_img * (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + img_squeeze.mean(dim=-2, keepdim=True)
                rec_img = rearrange(rec_img, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=14, w=14)

                # img = ToPILImage()(rec_img[0,:].clip(0,0.996))
                # img.save('rec_img.jpg')
                # pdb.set_trace()

                mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).cuda()[None, :, None, None]
                std = torch.as_tensor(IMAGENET_DEFAULT_STD).cuda()[None, :, None, None]
                rec_img = (rec_img - mean) / std
                im_q[mae_aug_idx] = rec_img

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._group_batch_shuffle_ddp(im_k, gpu_rank, node_rank, ngpu_per_node, nrank_per_subg, groups)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._group_batch_unshuffle_ddp(k, idx_unshuffle, gpu_rank, node_rank, ngpu_per_node, nrank_per_subg, groups)

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
        self._dequeue_and_enqueue(k)

        return logits, labels


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

