import numpy as np
import torch
import torch.nn as nn
import pdb

from easydict import EasyDict
from absl import flags
from absl import app 

FLAGS = flags.FLAGS

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

def compute_feat(model, loader, gpu_rank):
    num_feat = 0
    model.eval()
    if FLAGS.rank == 0:
        all_feats1 = np.zeros([FLAGS.dataset_len+1000, 2048]).astype(np.float32)
        all_feats_last = np.zeros([FLAGS.dataset_len+1000, FLAGS.moco_dim]).astype(np.float32)
        all_index = np.zeros([FLAGS.dataset_len+1000]).astype(np.int)
    
    for i, (images, target, index) in enumerate(loader):
        images = images.cuda(gpu_rank, non_blocking=True)
        index = index.cuda(gpu_rank, non_blocking=True)
        with torch.no_grad():
            k1, k_last = model.module.encoder_k(images, out_c4=True)
            k1 = nn.functional.normalize(k1, dim=1)
            k_last = nn.functional.normalize(k_last, dim=1)

        k1 = concat_all_gather(k1)
        k1 = k1.cpu().numpy()

        k_last = concat_all_gather(k_last)
        k_last = k_last.cpu().numpy()

        index = concat_all_gather(index)
        index = index.cpu().numpy()

        if i < len(loader) - 1:
            bsz = k1.shape[0]
            if FLAGS.rank == 0:
                all_feats1[i*bsz: (i+1)*bsz] = k1
                all_feats_last[i*bsz: (i+1)*bsz] = k_last
                all_index[i*bsz: (i+1)*bsz] = index
                num_feat += bsz
        else:
            if FLAGS.rank == 0:
                all_feats1[i*bsz: i*bsz + k1.shape[0]] = k1
                all_feats_last[i*bsz: i*bsz + k1.shape[0]] = k_last
                all_index[i*bsz: i*bsz + k1.shape[0]] = index
                num_feat += k1.shape[0]

        if i%200 == 0:
            print('%d | %d'%(i, len(loader)))

    print('num_feat: ', num_feat)
    model.train()
    if FLAGS.rank == 0:
        all_feats1 = all_feats1[:num_feat]
        all_feats_last = all_feats_last[:num_feat]
        all_index = all_index[:num_feat]

        sorted_index, sort_id = np.unique(all_index, return_index=True)
        sorted_feats1 = all_feats1[sort_id]
        sorted_feats_last = all_feats_last[sort_id]
        assert (all_index[sort_id] == np.arange(0, FLAGS.dataset_len)).all()

        return sorted_feats1, sorted_feats_last
    else:
        return 0, 0



def knn_kmeans(feat):
    if FLAGS.moxing:
        import faiss
    else:
        import mkl 
        mkl.get_max_threads()
        import faiss

    pseudo_label, distance, cluster_center = run_kmeans(feat)
    row_sum = np.linalg.norm(feat, axis=1)
    feat = feat / row_sum[:, np.newaxis]

    d = feat.shape[1]
    cpu_index = faiss.IndexFlatL2(d)
    index = faiss.index_cpu_to_all_gpus(cpu_index)
    index.add(feat)

    D, I = index.search(feat, FLAGS.knn_topk) # self-image is include in I[:,0]
    dis_index = np.argsort(np.array(distance)[I[:,1:]])
    imgs_corr = [[] for i in range(FLAGS.dataset_len)]

    for i in range(FLAGS.dataset_len):
        pos_num = 0 
        for idx in dis_index[i]:
            if pseudo_label[I[i,idx]] == pseudo_label[i]:
                pos_num += 1
                imgs_corr[i].append(I[i,idx])
            if pos_num == FLAGS.clus_pos_num:
                break

        if pos_num < FLAGS.clus_pos_num:
            imgs_corr[i].extend(I[i, 1:(FLAGS.clus_pos_num-pos_num+1)])

    imgs_corr = np.array(imgs_corr) # 1281167*FLAGS.clus_pos_num ndarray
    return EasyDict(imgs_corr=imgs_corr)



def knn(feat):
    if FLAGS.moxing:
        import faiss
    else:
        import mkl
        mkl.get_max_threads()
        import faiss
    d = feat.shape[1]
    cpu_index = faiss.IndexFlatL2(d)
    index = faiss.index_cpu_to_all_gpus(cpu_index)
    # index = cpu_index # only of debug
    index.add(feat)

    D, I = index.search(feat, FLAGS.clus_pos_num + 1) # self-image is include in I[:,0]
    imgs_corr = [[] for i in range(FLAGS.dataset_len)]
    for i in range(FLAGS.dataset_len):
        for j in range(FLAGS.clus_pos_num):
            imgs_corr[i].append(I[i,j+1])

    imgs_corr = np.array(imgs_corr) # 1281167*FLAGS.clus_pos_num ndarray
    return EasyDict(imgs_corr=imgs_corr)



def run_kmeans(feat):
    if FLAGS.moxing:
        import faiss
    else:
        import mkl
        mkl.get_max_threads()
        import faiss

    n_data, d = feat.shape
    '''
    # PCA preprocessing
    mat = faiss.PCAMatrix (d, 128, eigen_power=-0.5)
    mat.train(feat)
    feat = mat.apply_py(feat)
    d = 128
    # L2 normalization
    row_sums = np.linalg.norm(feat, axis=1)
    feat = feat / row_sums[:, np.newaxis]
    '''

    # faiss implementation of k-means
    clus = faiss.Clustering(d, FLAGS.cluster_K)
    clus.niter = 30
    clus.spherical = True
    clus.max_points_per_centroid = 10000000
    clus.seed = np.random.randint(12345)
    
    res = [faiss.StandardGpuResources() for i in range(FLAGS.ngpu)]
    flat_config = []
    for i in range(FLAGS.ngpu):
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = i
        flat_config.append(cfg)


    indexes = [faiss.GpuIndexFlatL2(res[i], d, flat_config[i]) for i in range(FLAGS.ngpu)]
    index = faiss.IndexReplicas()
    for sub_index in indexes:
        index.addIndex(sub_index)

    # index = faiss.IndexFlatL2(d) # only for debug

    # perform the training
    clus.train(feat, index)
    distance, I = index.search(feat, 1)
    print('mean distance: ', np.mean(distance))

    return [int(n[0]) for n in I], [d[0] for d in distance], faiss.vector_to_array(clus.centroids).reshape(-1, d)


def sta_kmeans(pseudo_label, distance):
    '''
    Args:
        pseudo_label: [531,531,0,...,6]      (1281167 len list)
        distance:     [0.1,0.1,0.03,...,0.2] (1281167 len list)
    '''
    imgs_per_pslabel = [[] for i in range(FLAGS.cluster_K)] # [[123, 221, 120], ..., [32, 987]]  (10000 len list)
    dis_per_pslabel = [[] for i in range(FLAGS.cluster_K)]  # [[0.1, 0.2, 0.1], ..., [0., 0.04]] (10000 len list)
    for i in range(FLAGS.dataset_len):
        imgs_per_pslabel[pseudo_label[i]].append(i)
        dis_per_pslabel[pseudo_label[i]].append(distance[i])
    mean_dist = [np.mean(tmp) for tmp in dis_per_pslabel]

    imgs_pos_flag = [[] for i in range(FLAGS.dataset_len)] # [1,1,0,1,...,1] (1 means close to center, 0 means far to center)
    for i in range(FLAGS.dataset_len):
        if distance[i] > mean_dist[pseudo_label[i]]: # use mean distance as thre
            imgs_pos_flag[i] = 0 if FLAGS.filter_kmeans else 1
        else:
            imgs_pos_flag[i] = 1


    pos_per_pseudo_label = [[] for i in range(FLAGS.cluster_K)] # select from imgs_per_pslabel by mean_dist
    for i in range(FLAGS.cluster_K):
        for j in range(len(imgs_per_pslabel[i])):
            if FLAGS.filter_kmeans:
                if dis_per_pslabel[i][j] <= mean_dist[i]:
                    pos_per_pseudo_label[i].append(imgs_per_pslabel[i][j])
            else:
                pos_per_pseudo_label[i].append(imgs_per_pslabel[i][j])


    # imgs_corr: 
    # for each sample i in dataset
    # 1) if imgs_pos_flag = 0, imgs_corr[i] = [i]
    # 2) if imgs_pos_flag = 1, imgs_corr[i] = [x,x,x] (img idx of same pseudo label with true imgs_pos_flag)
    imgs_corr = [[] for i in range(FLAGS.dataset_len)] 
    for i in range(FLAGS.dataset_len):
        if imgs_pos_flag[i] == 0:
            imgs_corr[i].append(i)
        else:
            imgs_corr[i].extend(pos_per_pseudo_label[pseudo_label[i]])


    # imgs_corr_format (formated imgs_corr):
    # for each sample i in imgs_corr 
    # 1) if len(imgs_corr[i]) < npos_per_pseudo_label, random.choice with replace
    # 2) if len(imgs_corr[i]) > npos_per_pseudo_label, random.choice without replace
    # with replace maens can select one multi times
    imgs_corr_format = [[] for i in range(FLAGS.dataset_len)]
    npos_per_pseudo_label = FLAGS.clus_pos_num
    for i in range(FLAGS.dataset_len):
        index = np.random.choice(len(imgs_corr[i]),
                                npos_per_pseudo_label,
                                replace=(len(imgs_corr[i]) <= npos_per_pseudo_label)
                                )
        imgs_corr_format[i].append([imgs_corr[i][j] for j in index])



    return np.array(imgs_corr_format).squeeze(1), np.array(imgs_pos_flag)

def kmeans(feat):
    pseudo_label, distance, cluster_center = run_kmeans(feat)
    imgs_corr, imgs_pos_flag = sta_kmeans(pseudo_label, distance)
    return EasyDict(imgs_corr=imgs_corr, pseudo_label=pseudo_label, 
            imgs_pos_flag=imgs_pos_flag, cluster_center=cluster_center)

