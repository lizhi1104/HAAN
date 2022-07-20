import torch
import torch.nn as nn
import torch.nn.init as torch_init
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import sys
import copy
import json
import pickle
import numpy as np
from PyTorchConv3D.models.i3d import InceptionI3D, Unit3D
from itertools import chain
from sklearn.cluster import KMeans
from collections import Counter


DEVICE = torch.device('cuda:2') if torch.cuda.is_available() else torch.device("cpu")
NUM_COARSE_CLASS = 14
NUM_FINE_CLASS = 106
MAX_T = 100
BATCH_SIZE = 2

threshold = 0
kmax_factor = 8
alpha = -0.5
beta = 0.3
epochs = 20
torch.manual_seed(0)

class training_set(Dataset):
    def __init__(self, features):
        vnames = np.load('annotations/FineAction/videoname.npy', allow_pickle=True)
        subset = np.load('annotations/FineAction/subset.npy', allow_pickle=True)
        labels_unique = np.load('annotations/FineAction/labels_unique.npy', allow_pickle=True)
        coarse_labels_unique = np.array([list(set(np.array(x)[:,0])) for x in labels_unique])
        fine_labels_unique = np.array([list(np.array(x)[:,1]) for x in labels_unique])
        labels = np.load('annotations/FineAction/labels.npy', allow_pickle=True)
        labels = np.array([list(np.array(x)[:,1]) for x in labels], dtype=object)
        segments = np.load('annotations/FineAction/segments.npy', allow_pickle=True)
        factors = np.load('annotations/FineAction/factors_100.npy', allow_pickle=True)
        
        self.features = features
        self.vnames = []
        self.seq_lens = []
        self.coarse_labels = []
        self.fine_labels = []
        self.factors = []
        self.truth_masks = []
        self.coarse_pos_counts = {i: 0 for i in range(NUM_COARSE_CLASS)}
        self.coarse_neg_counts = {i: 0 for i in range(NUM_COARSE_CLASS)}
        self.fine_pos_counts = {i: 0 for i in range(NUM_FINE_CLASS)}
        self.fine_neg_counts = {i: 0 for i in range(NUM_FINE_CLASS)}
        
        for i in range(len(vnames)):
            if subset[i] != 'training':
                continue
            self.vnames.append(vnames[i])
            self.seq_lens.append(len(features[i]))
            self.coarse_labels.append(self.convert_label(coarse_labels_unique[i], NUM_COARSE_CLASS))
            self.fine_labels.append(self.convert_label(fine_labels_unique[i], NUM_FINE_CLASS))
            self.factors.append(factors[i])
            self.truth_masks.append(self.get_truth_mask(labels[i], segments[i], [len(features[i]), NUM_FINE_CLASS], factors[i]))
            for j in range(NUM_COARSE_CLASS):
                if j in coarse_labels_unique[i]:
                    self.coarse_pos_counts[j] += 1
                else:
                    self.coarse_neg_counts[j] += 1
            for j in range(NUM_FINE_CLASS):
                if j in fine_labels_unique[i]:
                    self.fine_pos_counts[j] += 1
                else:
                    self.fine_neg_counts[j] += 1
        
    def __getitem__(self, index):
        vname = self.vnames[index]
        feature = self.features[index]
        seq_len = self.seq_lens[index]
        coarse_label = self.coarse_labels[index]
        fine_label = self.fine_labels[index]
        truth_mask = self.truth_masks[index]
        
        coarse_weights = np.zeros(NUM_COARSE_CLASS)
        for i in range(NUM_COARSE_CLASS):
            if coarse_label[i] == 1:
                coarse_weights[i] = NUM_COARSE_CLASS / self.coarse_pos_counts[i]
            else:
                coarse_weights[i] = NUM_COARSE_CLASS / self.coarse_neg_counts[i]
                
        fine_weights = np.zeros(NUM_FINE_CLASS)
        for i in range(NUM_FINE_CLASS):
            if fine_label[i] == 1:
                fine_weights[i] = NUM_FINE_CLASS / self.fine_pos_counts[i]
            else:
                fine_weights[i] = NUM_FINE_CLASS / self.fine_neg_counts[i]

        return vname, feature, seq_len, coarse_label, fine_label, coarse_weights, fine_weights, truth_mask

    def __len__(self):
        return len(self.vnames)
    
    def get_truth_mask(self, labels, segments, shape, factor):
        assert len(labels) == len(segments)
        mask = torch.zeros(shape)
        for i in range(len(labels)):
            start, end = segments[i]
            start_clip = int(round(start * factor))
            end_clip = min([int(round(end * factor)) + 1, shape[0]])
            mask[start_clip:end_clip,labels[i]] = 1
        return self.pad(mask)
    
    def convert_label(self, label, length):
        y = torch.zeros(length)
        y[label] = 1
        return y
    
    def pad(self, x):
        return np.pad(x, ((0, MAX_T-x.shape[0]), (0,0)), mode='constant', constant_values=0)
    
class test_set(Dataset):
    def __init__(self, features):
        vnames = np.load('annotations/FineAction/videoname.npy', allow_pickle=True)
        subset = np.load('annotations/FineAction/subset.npy', allow_pickle=True)
        labels = np.load('annotations/FineAction/labels.npy', allow_pickle=True)
        labels = np.array([list(np.array(x)[:,1]) for x in labels], dtype=object)
        segments = np.load('annotations/FineAction/segments.npy', allow_pickle=True)
        factors = np.load('annotations/FineAction/factors_100.npy', allow_pickle=True)
        
        self.features = features
        self.vnames = []
        self.labels = []
        self.segments = []
        self.factors = []
        
        for i in range(len(vnames)):
            if subset[i] != 'validation':
                continue
            self.vnames.append(vnames[i])
            self.labels.append(np.array(labels[i]))
            self.segments.append(np.array(segments[i]))
            self.factors.append(factors[i])
        
    def __getitem__(self, index):
        vname = self.vnames[index]
        feature = self.features[index]
        label = self.labels[index]
        segments = self.segments[index]
        factor = self.factors[index]
        
        return vname, feature, label, segments, factor

    def __len__(self):
        return len(self.vnames)
    
class Model(nn.Module):
    def __init__(self, n_dim, n_out, hidden_layer, dropout_rate):
        super(Model, self).__init__()
        hidden_sizes = hidden_layer.split(",")
        hidden_sizes = [int(size) for size in hidden_sizes]
        self.n_layers = len(hidden_sizes)
        in_dim = n_dim
        for i, hidden_size in enumerate(hidden_sizes):
            setattr(self, f"fc_{i+1}", nn.Linear(in_dim, hidden_size))
            in_dim = hidden_size
        self.out_layer = nn.Linear(in_dim, n_out)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs, is_training=True):
        x = inputs
        for i in range(self.n_layers):
            x = getattr(self, f"fc_{i+1}")(x)
            x = F.relu(x)
        if is_training:
            x = self.dropout(x)
        return x, self.out_layer(x)

def get_logits_agg_threshold(logits, threshold):
    mask = logits > threshold
    mask[torch.argmax(logits, dim=0), torch.arange(mask.size(1))] = True
    logits_agg = torch.sum(torch.mul(logits, mask), dim=0) / torch.sum(mask, dim=0)
    return logits_agg

def get_logits_agg_above_average(logits):
    average = torch.mean(logits, dim=0, keepdim=True)
    mask = logits >= average
    for i in range(mask.shape[1]):
        if not torch.any(mask[:,i]):
            mask[:,i][mask[:,i] == False] = True
    logits_agg = torch.sum(torch.mul(logits, mask), dim=0) / torch.sum(mask, dim=0)
    return logits_agg

def get_logits_agg_above_average_alpha(logits, alpha):
    ranges = torch.max(logits, dim=0)[0] - torch.min(logits, dim=0)[0]
    average = torch.mean(logits, dim=0, keepdim=True)
    mask = logits >= (average + alpha * ranges)
    mask[torch.argmax(logits, dim=0), torch.arange(mask.size(1))] = True
    logits_agg = torch.sum(torch.mul(logits, mask), dim=0) / torch.sum(mask, dim=0)
    return logits_agg

def get_logits_agg_above_middle(logits):
    ranges = torch.max(logits, dim=0)[0] - torch.min(logits, dim=0)[0]
    middle = ranges * 0.5 + torch.min(logits, dim=0)[0]
    mask = logits > middle
    logits_agg = torch.sum(torch.mul(logits, mask), dim=0) / torch.sum(mask, dim=0)
    return logits_agg

def get_logits_agg_above_middle_alpha(logits, alpha):
    ranges = torch.max(logits, dim=0)[0] - torch.min(logits, dim=0)[0]
    middle = ranges * 0.5 + torch.min(logits, dim=0)[0]
    mask = logits > (middle + alpha * ranges)
    logits_agg = torch.sum(torch.mul(logits, mask), dim=0) / torch.sum(mask, dim=0)
    return logits_agg

def get_logits_agg_kmax(logits, kmax_factor):
    seq_len = logits.size(0)
    k = max([1, int(seq_len / kmax_factor)])
    logits_agg = torch.mean(torch.topk(logits, k=k, dim=0)[0], dim=0)
    return logits_agg

def get_logits_agg_softmax_temporal(logits):
    attention = F.softmax(logits, dim=0)
    logits_agg = torch.mean(torch.mul(logits, attention), dim=0)
    return logits_agg

def get_logits_agg_softmax_class_temporal(logits):
    attention = F.softmax(logits, dim=1)
    attention = F.softmax(attention, dim=0)
    logits_agg = torch.mean(torch.mul(logits, attention), dim=0)
    return logits_agg

def get_features_agg_above_average(features, logits):
    n_clip, n_dim, n_vc = features.shape[0], features.shape[1], logits.shape[1]
    average = torch.mean(logits, dim=0, keepdim=True)
    mask = logits >= average
    for i in range(mask.shape[1]):
        if not torch.any(mask[:,i]):
            mask[:,i][mask[:,i] == False] = True
    mask_expand = mask.unsqueeze(2).expand(n_clip, n_vc, n_dim)
    features_expand = features.unsqueeze(1).expand(n_clip, n_vc, n_dim)
    features_agg = torch.sum(torch.mul(features_expand, mask_expand), dim=0) / torch.sum(mask, dim=0).unsqueeze(1)
    return features_agg

def get_features_agg_kmax(features, logits, kmax_factor):
    # featues: T * D
    # logits: T * M
    seq_len = logits.size(0)
    k = max([1, int(seq_len / kmax_factor)])
    indices = torch.topk(logits, k=k, dim=0)[1]
    return features[indices].mean(dim=0)

def get_true_position_values(logits, truth_masks, seq_lens):
    value = 0
    for i in range(len(seq_lens)):
        true_value = torch.mul(logits[i,:seq_lens[i]], truth_masks[i,:seq_lens[i]])
        true_value = torch.mean(true_value[true_value != 0])
        false_value = torch.mul(logits[i,:seq_lens[i]], 1 - truth_masks[i,:seq_lens[i]])
        false_value = torch.mean(false_value[false_value != 0])
        value = value + true_value - false_value
    value = value / len(seq_lens)
    return value.detach().cpu().item()

def get_predictions_mod_kmax(predictions, kmax_factor):
    predictions_mod = []
    c_score = []
    for p in predictions:
        pp = -p; [pp[:,i].sort() for i in range(pp.shape[1])]; pp = -pp
        c_s = np.mean(pp[:int(np.ceil(pp.shape[0] / kmax_factor)),:],axis=0)
        ind = c_s > 0.0
        c_score.append(c_s)
        new_pred = np.zeros(pp.shape, dtype='float32')
        predictions_mod.append(p * ind)
    return predictions_mod, c_score

def convert_label(label):
    label = list(set(np.array(label)))
    y = np.zeros(NUM_FINE_CLASS)
    for l in label:
        y[l] = 1
    return y

def getAP(conf, labels):
    assert len(conf) == len(labels)
    sortind = np.argsort(-conf)
    tp = labels[sortind] == 1
    fp = labels[sortind] != 1
    npos = np.sum(labels)
    
    fp = np.cumsum(fp).astype('float32')
    tp = np.cumsum(tp).astype('float32')
    rec = tp / npos
    prec = tp / (fp + tp)
    tmp = (labels[sortind] == 1).astype('float32')

    return np.sum(tmp * prec) / npos

def getClassificationMAP(confidence, labels):
    ''' confidence and labels are of dimension n_samples x n_label '''
    AP = []
    for i in range(labels.shape[1]):
        AP.append(getAP(confidence[:,i], labels[:,i]))
    return 100 * sum(AP) / len(AP)

def get_segment_gts(segments, labels):
    templabelidx = list(range(NUM_FINE_CLASS))
    segment_gts = {c: [] for c in templabelidx}
    for c in templabelidx:
        for i in range(len(segments)):
            for j in range(len(segments[i])):
                if labels[i][j] == c:
                    segment_gts[c].append([i, segments[i][j][0], segments[i][j][1]])
    return segment_gts

def get_segment_predictions(predictions, segments):
    templabelidx = list(range(NUM_FINE_CLASS))
    predictions, c_score = get_predictions_mod_kmax(predictions, kmax_factor)

    segment_predicts = {c: [] for c in templabelidx}
    
    for i in range(len(predictions)):
        segment_predict = []
        for c in templabelidx:
            tmp = predictions[i][:,c]
            threshold = np.mean(tmp) + (np.max(tmp) - np.min(tmp)) * alpha
            vid_pred = np.concatenate([np.zeros(1),(tmp > threshold).astype('float32'),np.zeros(1)], axis=0)
            vid_pred_diff = [vid_pred[idt] - vid_pred[idt - 1] for idt in range(1, len(vid_pred))]
            s = [idk for idk, item in enumerate(vid_pred_diff) if item==1]
            e = [idk for idk, item in enumerate(vid_pred_diff) if item==-1]
            for j in range(len(s)):
                aggr_score = np.max(tmp[s[j]: e[j]]) + beta * c_score[i][c]
                if e[j] - s[j] >= 1:               
                    segment_predict.append([i, s[j], e[j], aggr_score, c])
        if len(segment_predict) == 0:
            continue
        segment_predict = np.array(segment_predict)
        segment_predict = segment_predict[np.argsort(-segment_predict[:,3])]
        for predict in segment_predict:
            segment_predicts[predict[4]].append(predict[:4])

    for c in templabelidx:
        segment_predict = np.array(segment_predicts[c])
        if len(segment_predict) == 0:
            segment_predicts[c] = np.array([])
            continue
        segment_predicts[c] = segment_predict[np.argsort(-segment_predict[:,3])]
    return segment_predicts

def getLocMAP(segment_predicts, segment_gts, th, factors):
    ap = []
    for c in segment_predicts:
        segment_predict = segment_predicts[c]
        segment_gt = segment_gts[c]
        gtpos = len(segment_gt)
        if len(segment_predict) == 0:
            ap.append(0)
            continue
        tp, fp = [], []
        for i in range(len(segment_predict)):
            flag = 0.
            for j in range(len(segment_gt)):
                if segment_predict[i][0] == segment_gt[j][0]:
                    gtstart = int(round(segment_gt[j][1]) * factors[int(segment_predict[i][0])])
                    gtend = np.max([gtstart + 1, int(round(segment_gt[j][2] * factors[int(segment_predict[i][0])]))])
                    gt = range(gtstart, gtend)
                    p = range(int(segment_predict[i][1]), int(segment_predict[i][2]))
                    IoU = float(len(set(gt).intersection(set(p))))/float(len(set(gt).union(set(p))))
                    if IoU >= th:
                        flag = 1.
                        del segment_gt[j]
                        break
            tp.append(flag)
            fp.append(1. - flag)
        tp_c = np.cumsum(tp)
        fp_c = np.cumsum(fp)
        if sum(tp)==0:
            prc = 0.
        else:
            prc = np.sum((tp_c / (fp_c + tp_c)) * tp) / gtpos
        ap.append(prc)

    return 100 * np.mean(ap)

def getDetectionMAP(predictions, segments, labels, factors):
    iou_list = [0.5]
    dmap_list = []
    print('Getting predictions')
    sys.stdout.flush()
    segment_gts = get_segment_gts(segments, labels)
    segment_predicts = get_segment_predictions(predictions, segments)
    for iou in iou_list:
        print('Testing for IoU %f' %iou)
        sys.stdout.flush()
        dmap_list.append(getLocMAP(segment_predicts, copy.deepcopy(segment_gts), iou, factors))

    return dmap_list, iou_list


def generate_features(encoder):
    print("Generating new features")
    sys.stdout.flush()
    encoder.eval()
    vnames = np.load('annotations/videoname.npy', allow_pickle=True)
    subset = np.load('annotations/subset.npy', allow_pickle=True)
    features = []
    for i, (vname, sub) in enumerate(zip(vnames, subset)):
        if sub != 'training':
            continue
        i3d_feature = train_i3d_features[i]
        _, feature = encoder(torch.from_numpy(i3d_feature[None]).float().to(DEVICE))
        feature = feature.squeeze(0).detach().cpu().numpy()
        features.append(feature)
    features = np.array(features)
    return features
    
    
def generate_clustering_labels(features, num_vc, level, sample_rate=0.1):
    print("Generating clustering labels")
    sys.stdout.flush()
    vnames = np.load('annotations/FineAction/videoname.npy', allow_pickle=True)
    all_clips = np.array(list(chain(*features)))
    rand_indices = np.random.permutation(len(all_clips))[:int(sample_rate * len(all_clips))]
    sample_clips = all_clips[rand_indices]
    print(f"Fitting kmeans on {len(sample_clips)} clips")
    sys.stdout.flush()
    kmeans = KMeans(n_clusters=num_vc, random_state=0).fit(sample_clips)
    pickle.dump(kmeans, open(f"clustering/{level}_kmeans.pkl", "wb"))
    print(f"Inference on all clips")
    sys.stdout.flush()
    labels = kmeans.predict(all_clips)
    cur = 0
    os.makedirs(f"clustering/{level}_labels/", exist_ok=True)
    for vname, feature in zip(vnames, features):
        length = feature.shape[0]
        cur_labels = np.array(labels[cur:cur+length])
        np.save(f"clustering/{level}_labels/{vname}.npy", cur_labels)
        cur += length
    counts = Counter(labels)
    counts = {str(key): counts[key] for key in counts}
    json.dump(counts, open(f"clustering/{level}_label_counts.json", "w"), indent=2)
            

def run_test(fine_encoder, fine_classifier, test_loader, exp_name, epoch):
    fine_encoder.eval()
    fine_classifier.eval()
    instance_logits_stack = []
    element_logits_stack = []
    labels_unique_stack = []
    labels_stack = []
    segments_stack = []
    factors_stack = []
    for batch in test_loader:
        vname, feature, label, segments, factor = batch
        label = label.squeeze(0)
        segments = segments.squeeze(0)
        feature = feature.float().to(DEVICE)
        _, fine_feature = fine_encoder(feature, False)
        fine_feature = fine_feature.squeeze(0)
        _, fine_cls_logit = fine_classifier(fine_feature, False)
        fine_cls_logit_agg = get_logits_agg_above_average(fine_cls_logit)
        instance_logits_stack.append(fine_cls_logit_agg.detach().cpu().numpy())
        element_logits_stack.append(fine_cls_logit.detach().cpu().numpy())
        labels_unique_stack.append(convert_label(label))
        labels_stack.append(np.array(label))
        segments_stack.append(np.array(segments))
        factors_stack.append(factor.item())
    instance_logits_stack = np.array(instance_logits_stack)
    labels_unique_stack = np.array(labels_unique_stack)
    factors_stack = np.array(factors_stack)
    cmap = getClassificationMAP(instance_logits_stack, labels_unique_stack)
    dmap, iou = getDetectionMAP(element_logits_stack, segments_stack, labels_stack, factors_stack)
    print(f'Classification map {cmap:.3f}')
    for i, d in zip(iou, dmap):
        print(f'Detection map @ {i:.1f} = {d:.3f}')
    sys.stdout.flush()
    log = f"{epoch:03d} | {' | '.join([f'{x:.3f}' for x in dmap])} | {cmap:.3f}\n"
    with open(f"{logs_dir}/{exp_name}.txt", "a") as f:
        f.write(log)
    return dmap[0]
    
    
def get_closest_vc_for_class(classifier_weights, vc, vc_loss_method, top_vc_method, label, clustering_label):
    n_class, n_vc, n_dim = classifier_weights.shape[0], vc.shape[0], vc.shape[1]
    clustering_label = sorted(list(set(clustering_label.cpu().numpy())))
    non_clustering_label = [x for x in range(n_vc) if x not in clustering_label]
    if vc_loss_method == "cosine":
        temp_classifier_weights = classifier_weights.unsqueeze(1).expand(n_class, n_vc, n_dim).detach()
        temp_vc = vc.unsqueeze(0).expand(n_class, n_vc, n_dim).detach()
        sim = F.cosine_similarity(temp_classifier_weights, temp_vc, dim=2).detach()
        sim_valid = sim.clone()
        sim_valid[:,non_clustering_label] = -float('inf')
    elif vc_loss_method == "mse":
        sim = -torch.cdist(classifier_weights[None], vc[None]).squeeze(0).detach()
        sim_valid = sim.clone()
        sim_valid[:,non_clustering_label] = -float('inf')
    if "fixed" in top_vc_method:
        k = int(top_vc_method.split("_")[1])
        indices = torch.topk(sim, k=k, dim=1)[1]
        indices_valid = torch.topk(sim_valid, k=min([k, len(clustering_label)]), dim=1)[1]
        masks = torch.zeros(n_class, n_vc).to(DEVICE)
        for i, (index, index_valid) in enumerate(zip(indices, indices_valid)):
            if label[i] == 0:
                masks[i, index] = 1
            else:
                masks[i, index_valid] = 1
    elif "dis" in top_vc_method:
        k = float(top_vc_method.split("_")[1])
        dis = 1 - sim if vc_loss_method == "cosine" else -sim
        min_dis = torch.min(dis, dim=1)[0]
        pos_min_dis = min_dis * k
        masks = dis <= pos_min_dis.unsqueeze(1).expand(n_class, n_vc)
        dis_valid = 1 - sim_valid if vc_loss_method == "cosine" else -sim_valid
        min_dis_valid = torch.min(dis_valid, dim=1)[0]
        pos_min_dis_valid = min_dis_valid * k
        masks_valid = dis_valid <= pos_min_dis_valid.unsqueeze(1).expand(n_class, n_vc)
        masks[torch.where(label == 1)] = masks_valid[torch.where(label == 1)]
    return masks

def get_vc(cls_features, clustering_label, num_vc):
    dim = cls_features.shape[1]
    vc = torch.zeros((0, dim)).to(DEVICE)
    for i in range(num_vc):
        indices = torch.where(clustering_label == i)
        cur_vc = cls_features[indices]
        if cur_vc.shape[0] == 0:
            cur_vc = torch.zeros((1, dim)).to(DEVICE)
        else:
            cur_vc = cur_vc.mean(dim=0, keepdim=True)
        vc = torch.cat((vc, cur_vc), dim=0)
    return vc

def get_fine_vc(classifier_weights, vc, closest_vcs):
    n_class, n_vc, n_dim = classifier_weights.shape[0], vc.shape[0], vc.shape[1]
    k = closest_vcs.sum(dim=1).unsqueeze(1)
    temp_vc = vc.unsqueeze(0).expand(n_class, n_vc, n_dim)
    temp_closest_vcs = closest_vcs.unsqueeze(2).expand(n_class, n_vc, n_dim)
    return (temp_vc * temp_closest_vcs).sum(dim=1) / k

def get_vc_loss(classifier_weights, vc, labels, vc_loss_method):
    n_class, n_vc, n_dim = classifier_weights.shape[0], vc.shape[0], vc.shape[1]
    if vc_loss_method == "mse":
        mse = F.mse_loss(classifier_weights, vc, reduction="none").mean(dim=1)
        loss = mse[labels.bool()].sum() / labels.sum()
    elif vc_loss_method == "cosine":
        cos_dis = 1 - F.cosine_similarity(classifier_weights, vc, dim=1)
        loss = cos_dis[labels.bool()].sum() / labels.sum()
    return loss

def get_coarse_vc(fine_vc, coarse_to_fine_mapping, vc_fine_to_coarse_pool):
    dim = fine_vc.shape[1]
    coarse_vc = torch.zeros((0, dim)).to(DEVICE)
    for i in range(NUM_COARSE_CLASS):
        indices = coarse_to_fine_mapping[i]
        if vc_fine_to_coarse_pool == "mean":
            cur_vc = fine_vc[indices].mean(dim=0, keepdim=True)
        elif vc_fine_to_coarse_pool == "max":
            cur_vc = fine_vc[indices].max(dim=0, keepdim=True)[0]
        coarse_vc = torch.cat((coarse_vc, cur_vc), dim=0)
    return coarse_vc


train_i3d_features = np.load('i3d_100_train.npy', allow_pickle=True)
test_i3d_features = np.load('i3d_100_val.npy', allow_pickle=True)
training_loader = DataLoader(training_set(train_i3d_features), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set(test_i3d_features), batch_size=1, shuffle=False)

fine_to_coarse_mapping = json.load(open("annotations/FineAction/fine_to_coarse_mapping.json"))
fine_to_coarse_mapping = {int(k): v for k, v in fine_to_coarse_mapping.items()}
coarse_to_fine_mapping = {}
for f, c in fine_to_coarse_mapping.items():
    coarse_to_fine_mapping.setdefault(c, [])
    coarse_to_fine_mapping[c].append(f)

logs_dir = "logs_fineaction"
ckpt_dir = "ckpt_fineaction"
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

base_runs = ["1024_2048_0.25_above_average_0.0001",
             "1024_2048_0.5_above_average_0.0001",
             "1024_2048_0.3_above_average_0.0001"]
coarse_base_runs = os.listdir("ckpt_coarse")

hyperparameters = {
    "learning_rate": [0.00003, 0.00001],
    "num_fine_vc": [500],
    "clus_weight": [0.001],
    "clustering_loss_weight": [True],
    "vc_weight": [0.01],
    "vc_loss_method": ["cosine"],
    "top_vc_method": ["fixed_5"],
    "coarse_cls_weight": [1, 0.1, 0.01],
    "coarse_vc_weight": [0.1, 0.01, 0.001],
    "vc_fine_to_coarse_pool": ["mean"],
    "coarse_epoch": ["50"]
}
hyperparameter_keys = list(hyperparameters.keys())
parameter_lens = np.array([len(hyperparameters[key]) for key in hyperparameter_keys])

tried_sets = set()
for _ in range(9):
    while True:
        base_run_index = np.random.randint(len(base_runs))
        base_run = base_runs[base_run_index]
        base_run_parameters = base_run.split("_")
        feature_size = int(base_run_parameters[0])
        encoder_hidden_layer = base_run_parameters[1]
        dropout_rate = float(base_run_parameters[2])
        dropout_rate = 0
        mil_method = "_".join(base_run_parameters[3:5])
        base_learning_rate = float(base_run_parameters[5])
        coarse_base_run_index = np.random.randint(len(coarse_base_runs))
        coarse_base_run = coarse_base_runs[coarse_base_run_index]
        
        indices = np.floor(np.random.rand(len(parameter_lens)) * parameter_lens).astype(int)
        for i, key in enumerate(hyperparameter_keys):
            exec(f"{key}=hyperparameters[key][indices[i]]")
        
        parameter_set = tuple([base_run] + [hyperparameters[key][indices[i]] for i, key in enumerate(hyperparameter_keys)])
        exp_name = '_'.join([str(x) for x in parameter_set])
        if vc_loss_method == "mse":
            if top_vc_method in ["dis_1.01", "dis_1.02", "dis_1.05", "dis_1.1"]:
                continue
        elif vc_loss_method == "cosine":
            if top_vc_method in ["dis_1.5", "dis_2", "dis_5", "dis_10"]:
                continue
        if os.path.exists(f'{ckpt_dir}/{exp_name}/'):
            continue
        if parameter_set not in tried_sets:
            tried_sets.add(parameter_set)
            break

    print(exp_name)
    torch.manual_seed(0)
    np.random.seed(0)
    fine_encoder = nn.DataParallel(Model(4096, feature_size, encoder_hidden_layer, dropout_rate), device_ids=[2])
    fine_classifier = nn.DataParallel(Model(feature_size, NUM_FINE_CLASS, str(feature_size), dropout_rate), device_ids=[2])
    fine_self_sup_model = nn.DataParallel(Model(feature_size, num_fine_vc, str(feature_size), dropout_rate), device_ids=[2])
    coarse_encoder = nn.DataParallel(Model(4096, feature_size, encoder_hidden_layer, dropout_rate), device_ids=[2])
    coarse_classifier = nn.DataParallel(Model(feature_size, NUM_COARSE_CLASS, str(feature_size), dropout_rate), device_ids=[2])
    
    fine_encoder_pkl = f"ckpt_fine_only_base/{base_run}/fine_encoder_50.pkl"
    fine_classifier_pkl = f"ckpt_fine_only_base/{base_run}/fine_classifier_50.pkl"
    if coarse_epoch == "50":
        coarse_encoder_pkl = f"ckpt_coarse/{coarse_base_run}/coarse_encoder_50.pkl"
        coarse_classifier_pkl = f"ckpt_coarse/{coarse_base_run}/coarse_classifier_50.pkl"
    elif coarse_epoch == "best":
        coarse_encoder_pkl = f"ckpt_coarse/{coarse_base_run}/best_coarse_encoder.pkl"
        coarse_classifier_pkl = f"ckpt_coarse/{coarse_base_run}/best_coarse_classifier.pkl"
    fine_encoder.load_state_dict(torch.load(fine_encoder_pkl))
    fine_classifier.load_state_dict(torch.load(fine_classifier_pkl))
    coarse_encoder.load_state_dict(torch.load(coarse_encoder_pkl))
    coarse_classifier.load_state_dict(torch.load(coarse_classifier_pkl))
    
    parameters = list(fine_encoder.parameters()) + list(fine_classifier.parameters()) + list(fine_self_sup_model.parameters()) + list(coarse_encoder.parameters()) + list(coarse_classifier.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    output_dir = f'{ckpt_dir}/{exp_name}/'
    os.makedirs(output_dir, exist_ok=True)
    best_result = 0
    
    for e in range(epochs):
        print("="*20)
        print(f"epoch {e+1}")
        print("="*20)
        sys.stdout.flush()
        
        if e % 5 == 0:
            fine_features = generate_features(fine_encoder)
            generate_clustering_labels(fine_features, num_fine_vc, "fine", 0.03)

            fine_clustering_counts = json.load(open("clustering/fine_label_counts.json"))
            fine_clustering_total_counts = sum(fine_clustering_counts.values())
            fine_clustering_weights = [fine_clustering_counts[str(i)] / fine_clustering_total_counts for i in range(num_fine_vc)]
            fine_clustering_weights = torch.tensor(fine_clustering_weights).float().to(DEVICE)

        count = 0
        for batch in training_loader:
            vnames, features, seq_lens, coarse_labels, fine_labels, coarse_weights, fine_weights, truth_masks = batch

            features = features[:,:torch.max(seq_lens)].float().to(DEVICE)
            fine_encoder.train()
            fine_classifier.train()
            fine_self_sup_model.train()
            coarse_encoder.train()
            coarse_classifier.train()
            optimizer.zero_grad()

            _, fine_features = fine_encoder(features)
            fine_cls_features, fine_cls_logits = fine_classifier(fine_features)
            _, fine_clustering_logits = fine_self_sup_model(fine_features)
            _, coarse_features = coarse_encoder(features)
            _, coarse_cls_logits = coarse_classifier(coarse_features)
            
            fine_cls_loss = torch.zeros(1).to(DEVICE)
            fine_clustering_loss = torch.zeros(1).to(DEVICE)
            fine_vc_loss = torch.zeros(1).to(DEVICE)
            coarse_cls_loss = torch.zeros(1).to(DEVICE)
            coarse_vc_loss = torch.zeros(1).to(DEVICE)
            fine_classifier_weights = fine_classifier.module.out_layer.weight.detach()
            coarse_classifier_weights = coarse_classifier.module.out_layer.weight.detach()
            for j in range(len(seq_lens)):
                fine_cls_logit = fine_cls_logits[j,:seq_lens[j]]
                coarse_cls_logit = coarse_cls_logits[j,:seq_lens[j]]
                if mil_method == "above_average":
                    fine_cls_logit_agg = get_logits_agg_above_average(fine_cls_logit)
                    coarse_cls_logit_agg = get_logits_agg_above_average(coarse_cls_logit)
                elif "kmax" in mil_method:
                    k = int(mil_method.split("_")[1])
                    fine_cls_logit_agg = get_logits_agg_kmax(fine_cls_logit, k)
                    coarse_cls_logit_agg = get_logits_agg_kmax(coarse_cls_logit, k)
                fine_cls_loss += F.binary_cross_entropy_with_logits(fine_cls_logit_agg,
                                                                    fine_labels[j].to(DEVICE),
                                                                    weight=fine_weights[j].to(DEVICE))
                coarse_cls_loss += F.binary_cross_entropy_with_logits(coarse_cls_logit_agg,
                                                                      coarse_labels[j].to(DEVICE),
                                                                      weight=coarse_weights[j].to(DEVICE))
                
                vname = vnames[j]
                fine_clustering_logit = fine_clustering_logits[j,:seq_lens[j]]
                fine_clustering_label = np.load(f"clustering/fine_labels/{vname}.npy", allow_pickle=True)
                fine_clustering_label = torch.from_numpy(fine_clustering_label).long().to(DEVICE)
                if clustering_loss_weight:
                    fine_clustering_loss += F.cross_entropy(fine_clustering_logit,
                                                            fine_clustering_label,
                                                            weight=fine_clustering_weights) / seq_lens[j]
                else:
                    fine_clustering_loss += F.cross_entropy(fine_clustering_logit,
                                                            fine_clustering_label) / seq_lens[j]
                
                
                vc = get_vc(fine_cls_features[j,:seq_lens[j]], fine_clustering_label, num_fine_vc)
                fine_closest_vcs = get_closest_vc_for_class(fine_classifier_weights, vc, vc_loss_method, top_vc_method, fine_labels[j], fine_clustering_label)
                fine_vc = get_fine_vc(fine_classifier_weights, vc, fine_closest_vcs)
                coarse_vc = get_coarse_vc(fine_vc, coarse_to_fine_mapping, vc_fine_to_coarse_pool)
                
                fine_vc_loss += get_vc_loss(fine_classifier_weights, fine_vc, fine_labels[j], vc_loss_method)
                coarse_vc_loss += get_vc_loss(coarse_classifier_weights, coarse_vc, coarse_labels[j], vc_loss_method)
                
            loss = fine_cls_loss + clus_weight * fine_clustering_loss + vc_weight * fine_vc_loss + coarse_cls_weight * coarse_cls_loss + coarse_vc_weight * coarse_vc_loss

            print(f'batch {count:04d} loss={loss.item():.5f} cls_loss={fine_cls_loss.item():.5f} clustering_loss={fine_clustering_loss.item():.5f} vc_loss={fine_vc_loss.item():.5f} coarse_cls_loss={coarse_cls_loss.item():.5f} coarse_vc_loss={coarse_vc_loss.item():.5f}')  
            sys.stdout.flush()

            loss.backward()
            optimizer.step()
            count += 1

        print("="*20)
        if (e+1) % 10 == 0:
            print(f"epoch {e+1} ends")
            print("saving checkpoints")
            sys.stdout.flush()
            torch.save(fine_encoder.state_dict(), f'{output_dir}/fine_encoder_{e+1}.pkl')
            torch.save(fine_classifier.state_dict(), f'{output_dir}/fine_classifier_{e+1}.pkl')
            torch.save(fine_self_sup_model.state_dict(), f'{output_dir}/fine_self_sup_model_{e+1}.pkl')
            torch.save(coarse_encoder.state_dict(), f'{output_dir}/coarse_encoder_{e+1}.pkl')
            torch.save(coarse_classifier.state_dict(), f'{output_dir}/coarse_classifier_{e+1}.pkl')
            print("checkpoints saved")
            sys.stdout.flush()
        dmap = run_test(fine_encoder, fine_classifier, test_loader, exp_name, e+1)
        if dmap > best_result:
            torch.save(fine_encoder.state_dict(), f'{output_dir}/best_fine_encoder.pkl')
            torch.save(fine_classifier.state_dict(), f'{output_dir}/best_fine_classifier.pkl')
            torch.save(fine_self_sup_model.state_dict(), f'{output_dir}/best_fine_self_sup_model.pkl')
            torch.save(coarse_encoder.state_dict(), f'{output_dir}/best_coarse_encoder.pkl')
            torch.save(coarse_classifier.state_dict(), f'{output_dir}/best_coarse_classifier.pkl')
            best_result = dmap
        
