import numpy as np
from torch.utils.data import DataLoader, Dataset


class DatasetManager:
    @classmethod
    def get_dataset_loaders(cls, configs):
        features = np.load(configs["features"]["features_path"], allow_pickle=True)
        annotations = {
            "vnames": np.load(
                configs["annotations"]["video_names_path"], allow_pickle=True
            ),
            "labels": np.load(configs["annotations"]["labels_path"], allow_pickle=True),
            "segments": np.load(
                configs["annotations"]["segments_path"], allow_pickle=True
            ),
            "sec2clip_ratios": np.load(
                configs["annotations"]["sec2clip_ratios_path"], allow_pickle=True
            ),
        }

        training_loader = DataLoader(
            TrainingSet(configs["stats"], features, annotations),
            batch_size=configs["hyperparameters"]["batch_size"],
            shuffle=True,
        )
        test_loader = DataLoader(
            TestSet(configs["stats"], features, annotations),
            batch_size=1,
            shuffle=False,
        )

        return training_loader, test_loader


class TrainingSet(Dataset):
    def __init__(self, stats, features, annotations):
        num_training_videos = stats["num_training_videos"]
        self.num_videos = num_training_videos

        self.num_coarse_classes = stats["num_coarse_classes"]
        self.num_fine_classes = stats["num_fine_classes"]
        self.features = [
            self._pad(features[:num_training_videos][i], stats["max_clips_per_video"])
            for i in range(self.num_videos)
        ]
        self.vnames = annotations["vnames"][:num_training_videos]
        self.labels = annotations["labels"][:num_training_videos]
        self.seq_lens = [
            len(features[:num_training_videos][i]) for i in range(self.num_videos)
        ]

        self.coarse_labels = np.array(
            [
                self._encode_labels(np.array(x)[:, 0], self.num_coarse_classes)
                for x in self.labels
            ]
        )
        self.fine_labels = np.array(
            [
                self._encode_labels(np.array(x)[:, 1], self.num_fine_classes)
                for x in self.labels
            ]
        )

        coarse_pos_counts = self.coarse_labels.sum(axis=0)
        coarse_neg_counts = (
            np.ones(self.num_coarse_classes) * self.num_videos - coarse_pos_counts
        )
        fine_pos_counts = self.fine_labels.sum(axis=0)
        fine_neg_counts = (
            np.ones(self.num_fine_classes) * self.num_videos - fine_pos_counts
        )

        self.coarse_pos_weight = self.num_coarse_classes / coarse_pos_counts
        self.coarse_neg_weight = self.num_coarse_classes / coarse_neg_counts
        self.fine_pos_weight = self.num_fine_classes / fine_pos_counts
        self.fine_neg_weight = self.num_fine_classes / fine_neg_counts

    def __getitem__(self, index):
        vname = self.vnames[index]
        feature = self.features[index]
        seq_len = self.seq_lens[index]
        coarse_label = self.coarse_labels[index]
        fine_label = self.fine_labels[index]
        coarse_weights = np.array([self.coarse_neg_weight, self.coarse_pos_weight])[
            (coarse_label.astype(int), np.arange(coarse_label.shape[0]))
        ]
        fine_weights = np.array([self.fine_neg_weight, self.fine_pos_weight])[
            (fine_label.astype(int), np.arange(fine_label.shape[0]))
        ]
        return (
            vname,
            feature,
            seq_len,
            coarse_label,
            fine_label,
            coarse_weights,
            fine_weights,
        )

    def __len__(self):
        return self.num_videos

    def _encode_labels(self, label, length):
        y = np.zeros(length)
        y[label] = 1
        return y

    def _pad(self, x, max_t):
        return np.pad(
            x, ((0, max_t - x.shape[0]), (0, 0)), mode="constant", constant_values=0
        )


class TestSet(Dataset):
    def __init__(self, stats, features, annotations):
        num_training_videos = stats["num_training_videos"]
        self.num_videos = features.shape[0] - num_training_videos

        self.features = features[num_training_videos:]
        self.vnames = annotations["vnames"][num_training_videos:]
        self.labels = np.array(
            [
                list(np.array(x)[:, 1])
                for x in annotations["labels"][num_training_videos:]
            ],
            dtype=object,
        )
        self.segments = annotations["segments"][num_training_videos:]
        self.sec2clip_ratios = annotations["sec2clip_ratios"][num_training_videos:]

    def __getitem__(self, index):
        vname = self.vnames[index]
        feature = self.features[index]
        label = np.array(self.labels[index])
        segments = np.array(self.segments[index])
        sec2clip_ratio = self.sec2clip_ratios[index]

        return vname, feature, label, segments, sec2clip_ratio

    def __len__(self):
        return self.num_videos
