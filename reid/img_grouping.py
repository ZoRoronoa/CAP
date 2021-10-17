import torch
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.cluster.dbscan_ import dbscan
from sklearn.cluster import KMeans
from torch._C import device
from reid.utils.rerank import compute_jaccard_dist
from reid.utils.faiss_rerank import faiss_compute_jaccard_dist
import scipy.io as sio 
torch.autograd.set_detect_anomaly(True)


def img_association(network, propagate_loader, min_sample=4, eps=0,
                    rerank=False, k1=20, k2=6, intra_id_reinitialize=False):

    network.eval()
    print('Start Inference...')
    features = []
    global_labels = []
    all_cams = []
    real_labels = []
    with torch.no_grad():
        for c, data in enumerate(propagate_loader):
            # changes here
            
            images = data[0]
            r_label = data[2]
            g_label = data[3]
            # print("data1 & data2 be like: ", data[1], data[2])
            # print("g_label: {}", g_label)
            cam = data[4]
            images = images.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            embed_feat = network(images)
            features.append(embed_feat.cpu())

            real_labels.append(r_label)
            global_labels.append(g_label)
            all_cams.append(cam)

    features = torch.cat(features, dim=0).numpy()
    real_labels = torch.cat(real_labels, dim=0).numpy()
    global_labels = torch.cat(global_labels, dim=0).numpy()
    all_cams = torch.cat(all_cams, dim=0).numpy()
    # print('  features: shape= {}'.format(features.shape))
    # print('  feature be like', features, type(features))
    # print('  real_labels be like', real_labels, type(real_labels))
    # print('  global_labels be like', global_labels, type(global_labels))
    # print('  all_cams be like', all_cams, type(all_cams))
    # if needed, average camera-style transferred image features

    new_features = []
    new_cams = []
    for glab in np.unique(global_labels):
        idx = np.where(global_labels == glab)[0]
        new_features.append(np.mean(features[idx], axis=0))
        new_cams.append(all_cams[idx])

    new_features = np.array(new_features)
    new_cams = np.array(new_cams).squeeze()
    # print("new_features be like: ", new_features)
    # print("new_cams be like: ", new_cams)
    del features, all_cams
    # eps: ⽤于设置密度聚类中的e领域
    # compute distance W
    new_features = new_features / np.linalg.norm(new_features, axis=1, keepdims=True)  # l2-normalize
    if rerank:
        W = faiss_compute_jaccard_dist(torch.from_numpy(new_features), k1=k1, k2=k2)
    else:
        W = cdist(new_features, new_features, 'euclidean')
    print('  distance matrix: shape= {}'.format(W.shape))

    # self-similarity for association
    print('  perform image grouping...')
    # changes here
    # _, updated_label = dbscan(W, eps=eps, min_samples=min_sample, metric='precomputed', n_jobs=8)
    updated_label = real_labels
    print("W be like", W)
    # print("updated_label be like", updated_label)
                                                                                                                                         
    print('  eps in cluster: {:.3f}'.format(eps))
    print('  updated_label: num_class= {}, {}/{} images are associated.'
          .format(updated_label.max() + 1, len(updated_label[updated_label >= 0]), len(updated_label)))

    if intra_id_reinitialize:
        print('re-computing initialized intra-ID feature...')
        intra_id_features = []
        intra_id_labels = []
        for cc in np.unique(new_cams):
            percam_ind = np.where(new_cams == cc)[0]
            percam_feature = new_features[percam_ind, :]
            percam_label = updated_label[percam_ind]
            percam_class_num = len(np.unique(percam_label[percam_label >= 0]))
            percam_id_feature = np.zeros((percam_class_num, percam_feature.shape[1]), dtype=np.float32)
            cnt = 0
            for lbl in np.unique(percam_label):
                if lbl >= 0:
                    ind = np.where(percam_label == lbl)[0]
                    id_feat = np.mean(percam_feature[ind], axis=0)
                    percam_id_feature[cnt, :] = id_feat
                    intra_id_labels.append(lbl)
                    cnt += 1
            percam_id_feature = percam_id_feature / np.linalg.norm(percam_id_feature, axis=1, keepdims=True)
            intra_id_features.append(torch.from_numpy(percam_id_feature))
        return updated_label, intra_id_features

