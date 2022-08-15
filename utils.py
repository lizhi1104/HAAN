import sys
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


def get_aggregated_scores(scores):
    average = torch.mean(scores, dim=0, keepdim=True)
    mask = scores >= average
    return torch.sum(torch.mul(scores, mask), dim=0) / torch.sum(mask, dim=0)


def get_pseudo_labels(dataset_loader, encoder, num_clusters, sample_rate, device):
    log_message("Start generating pseudo labels...")
    vnames, seq_lens, all_clips = [], [], []
    encoder.eval()
    for batch in dataset_loader:
        vname, feature, seq_len, _, _, _, _ = batch
        vnames += vname
        seq_lens.append(seq_len)
        _, encoded_feature = encoder(feature.float().to(device))
        encoded_feature = encoded_feature.detach().cpu()
        for i in range(len(feature)):
            all_clips.append(encoded_feature[i][: seq_len[i]])
    vnames = np.array(vnames)
    seq_lens = torch.cat(seq_lens).numpy()
    all_clips = torch.cat(all_clips).numpy()

    sample_clips = _get_sample_clips(all_clips, sample_rate)
    log_message(f"Fitting kmeans on {len(sample_clips)} clips...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(sample_clips)

    log_message(f"Inference on all {len(all_clips)} clips...")
    labels = kmeans.predict(all_clips)
    video_labels = _get_video_labels(labels, vnames, seq_lens)
    label_weights = _get_pseudo_label_weights(labels, num_clusters, device)
    return video_labels, label_weights


def _get_sample_clips(clips, sample_rate):
    rand_indices = np.random.permutation(len(clips))[: int(sample_rate * len(clips))]
    return clips[rand_indices]


def _get_video_labels(clip_labels, vnames, seq_lens):
    video_labels, cur = {}, 0
    for vname, seq_len in zip(vnames, seq_lens):
        cur_labels = clip_labels[cur : cur + seq_len]
        video_labels[vname] = cur_labels
        cur += seq_len
    return video_labels


def _get_pseudo_label_weights(labels, num_clusters, device):
    counts = Counter(labels)
    total_counts = sum(counts.values())
    weights = [counts[i] / total_counts for i in range(num_clusters)]
    return torch.tensor(weights).float().to(device)


def get_visual_concepts_from_clip_features(
    feature, pseudo_label, num_visual_concepts, device
):
    dim = feature.shape[1]
    visual_concepts = []
    for i in range(num_visual_concepts):
        indices = torch.where(pseudo_label == i)
        vc = feature[indices]
        if vc.shape[0] == 0:
            vc = torch.zeros((1, dim)).to(device)
        else:
            vc = vc.mean(dim=0, keepdim=True)
        visual_concepts.append(vc)
    return torch.cat(visual_concepts)


def get_fine_level_reps_from_visual_concepts(
    classifier_weights, visual_concepts, top_k, device
):
    similarity = _calculate_cosine_similarity(classifier_weights, visual_concepts)
    closest_masks = _get_closest_visual_concepts_masks(similarity, top_k, device)
    fine_level_reps = _aggregate_visual_concepts(visual_concepts, closest_masks, top_k)
    return fine_level_reps


def _calculate_cosine_similarity(classifier_weights, visual_concepts):
    n_class, n_vc, n_dim = (
        classifier_weights.shape[0],
        visual_concepts.shape[0],
        visual_concepts.shape[1],
    )
    temp_cws = classifier_weights.unsqueeze(1).expand(n_class, n_vc, n_dim).detach()
    temp_vcs = visual_concepts.unsqueeze(0).expand(n_class, n_vc, n_dim).detach()
    return F.cosine_similarity(temp_cws, temp_vcs, dim=2).detach()


def _get_closest_visual_concepts_masks(similarity, top_k, device):
    n_class, n_vc = similarity.shape[0], similarity.shape[1]
    indices = torch.topk(similarity, k=top_k, dim=1)[1]
    closest_masks = torch.zeros(n_class, n_vc).to(device)
    for i, index in enumerate(indices):
        closest_masks[i, index] = 1
    return closest_masks


def _aggregate_visual_concepts(visual_concepts, closest_masks, top_k):
    n_class, n_vc, n_dim = (
        closest_masks.shape[0],
        closest_masks.shape[1],
        visual_concepts.shape[1],
    )
    temp_vcs = visual_concepts.unsqueeze(0).expand(n_class, n_vc, n_dim)
    temp_masks = closest_masks.unsqueeze(2).expand(n_class, n_vc, n_dim)
    return (temp_vcs * temp_masks).sum(dim=1) / top_k


def get_coarse_level_reps_from_fine_level_reps(
    fine_level_reps, coarse_to_fine_mappings, pooling_method
):
    dim = fine_level_reps.shape[1]
    num_coarse_classes = len(coarse_to_fine_mappings)
    coarse_level_reps = []
    for i in range(num_coarse_classes):
        indices = coarse_to_fine_mappings[i]
        if pooling_method == "mean":
            cur_reps = fine_level_reps[indices].mean(dim=0, keepdim=True)
        elif pooling_method == "max":
            cur_reps = fine_level_reps[indices].max(dim=0, keepdim=True)[0]
        else:
            raise Exception("Invalid pooling method!")
        coarse_level_reps.append(cur_reps)
    return torch.cat(coarse_level_reps)


def log_message(message):
    print(message)
    sys.stdout.flush()
