import os

import numpy as np
import pandas as pd

import utils
from eval_detection import compute_average_precision_detection


class Evaluator:
    def __init__(
        self,
        configs,
        test_loader,
        encoder,
        fine_level_classifier,
        output_dir,
        device,
        verbose=True,
    ):
        self.loader = test_loader
        self.encoder = encoder
        self.fine_level_classifier = fine_level_classifier
        self.device = device
        self.verbose = verbose

        self.num_classes = configs["stats"]["num_fine_classes"]
        self.tiou_thresholds = configs["evaluation"]["tiou_thresholds"]
        self.report_thresholds = configs["evaluation"]["report_thresholds"]
        self.alpha = configs["evaluation"]["alpha"]
        self.kmax_mod_factor = configs["evaluation"]["kmax_mod_factor"]

        self.segment_gts = None
        self.results_path = os.path.join(output_dir, "results.csv")
        self.results_header = (
            ["epoch"] + [f"mAP@{t}" for t in self.report_thresholds] + ["avg.mAP"]
        )
        self.results = []

    def evaluate(self, epoch_num=0):
        self._start_evaluation()

        scores, labels, segments, sec2clip_ratios = [], [], [], []
        for batch in self.loader:
            vname, feature, label, segment, sec2clip_ratio = batch

            feature = feature.float().to(self.device)
            _, encoded_feature = self.encoder(feature, is_training=False)
            encoded_feature = encoded_feature.squeeze(0)
            _, cls_score = self.fine_level_classifier(
                encoded_feature, is_training=False
            )

            scores.append(cls_score.detach().cpu().numpy())
            labels.append(np.array(label.squeeze(0)))
            segments.append(np.array(segment.squeeze(0)))
            sec2clip_ratios.append(sec2clip_ratio.item())

        segment_gts = self._get_segment_gts(segments, labels)
        segment_predictions = self._get_segment_predictions(scores, sec2clip_ratios)

        average_precisions = np.empty((self.num_classes, len(self.tiou_thresholds)))
        for i in range(self.num_classes):
            gt_df = pd.DataFrame(
                segment_gts[i], columns=["video-id", "t-start", "t-end"]
            )
            pr_df = pd.DataFrame(
                segment_predictions[i],
                columns=["video-id", "t-start", "t-end", "score"],
            )
            average_precisions[i] = compute_average_precision_detection(
                gt_df, pr_df, self.tiou_thresholds
            )

        mAP = dict(zip(self.tiou_thresholds, average_precisions.mean(axis=0) * 100))
        result = [epoch_num]
        for threshold in self.report_thresholds:
            result.append(mAP[threshold])
            utils.log_message(f"mAP@{threshold} = {round(mAP[threshold], 2)}")
        mAP_avg = average_precisions.mean(axis=0).mean() * 100
        result.append(mAP_avg)
        utils.log_message(f"average mAP = {round(mAP_avg, 2)}")
        self.results.append(result)
        self._save_results()

        return mAP_avg

    def _get_segment_gts(self, segments, labels):
        if self.segment_gts is not None:
            return self.segment_gts

        segment_gts = {c: [] for c in range(self.num_classes)}
        for c in range(self.num_classes):
            for i in range(len(segments)):
                for j in range(len(segments[i])):
                    if labels[i][j] == c:
                        segment_gts[c].append([i, segments[i][j][0], segments[i][j][1]])
        self.segment_gts = segment_gts

        return segment_gts

    def _get_segment_predictions(self, scores, sec2clip_ratios):
        scores, c_scores = self._get_scores_mod(scores)

        segment_predictions = {c: [] for c in range(self.num_classes)}

        for i in range(len(scores)):
            sec2clip_ratio = sec2clip_ratios[i]
            segment_prediction = []
            for c in range(self.num_classes):
                tmp = scores[i][:, c]
                threshold = np.mean(tmp) + (np.max(tmp) - np.min(tmp)) * self.alpha
                vid_pred = np.concatenate(
                    [np.zeros(1), (tmp > threshold).astype("float32"), np.zeros(1)],
                    axis=0,
                )
                vid_pred_diff = [
                    vid_pred[idt] - vid_pred[idt - 1] for idt in range(1, len(vid_pred))
                ]
                s = [idk for idk, item in enumerate(vid_pred_diff) if item == 1]
                e = [idk for idk, item in enumerate(vid_pred_diff) if item == -1]
                for j in range(len(s)):
                    aggr_score = np.max(tmp[s[j] : e[j]]) + 0.5 * c_scores[i][c]
                    if e[j] - s[j] >= 1:
                        segment_prediction.append(
                            [
                                i,
                                s[j] / sec2clip_ratio,
                                e[j] / sec2clip_ratio,
                                aggr_score,
                                c,
                            ]
                        )
            if len(segment_prediction) == 0:
                continue
            segment_prediction = np.array(segment_prediction)
            segment_prediction = segment_prediction[
                np.argsort(-segment_prediction[:, 3])
            ]
            for prediction in segment_prediction:
                segment_predictions[prediction[4]].append(prediction[:4])

        for c in range(self.num_classes):
            segment_prediction = np.array(segment_predictions[c])
            if len(segment_prediction) == 0:
                segment_predictions[c] = []
                continue
            segment_predictions[c] = segment_prediction[
                np.argsort(-segment_prediction[:, 3])
            ]

        return segment_predictions

    def _get_scores_mod(self, scores):
        scores_mod, c_scores = [], []
        for score in scores:
            sorted_score = -np.sort(-score, axis=0)
            c_s = np.mean(
                sorted_score[
                    : int(np.ceil(sorted_score.shape[0] / self.kmax_mod_factor)), :
                ],
                axis=0,
            )
            ind = c_s > 0.0
            c_scores.append(c_s)
            scores_mod.append(score * ind)
        return scores_mod, c_scores

    def _save_results(self):
        df = pd.DataFrame(self.results, columns=self.results_header)
        df.to_csv(self.results_path, index=False)

    def _start_evaluation(self):
        self.encoder.eval()
        self.fine_level_classifier.eval()
