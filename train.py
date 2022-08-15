import json

import torch
import torch.nn.functional as F

import utils


class Trainer:
    def __init__(
        self,
        configs,
        training_loader,
        encoder,
        fine_level_classifier,
        pseudo_label_classifier,
        coarse_level_classifier,
        evaluator,
        output_dir,
        device,
        verbose=True,
    ):
        self.loader = training_loader
        self.encoder = encoder
        self.fine_level_classifier = fine_level_classifier
        self.pseudo_label_classifier = pseudo_label_classifier
        self.coarse_level_classifier = coarse_level_classifier
        self.evaluator = evaluator
        self.device = device
        self.verbose = verbose
        self.output_dir = output_dir

        self.clustering_sample_rate = configs["hyperparameters"][
            "clustering_sample_rate"
        ]
        self.learning_rate = configs["hyperparameters"]["learning_rate"]
        self.epochs = configs["hyperparameters"]["epochs"]
        self.num_visual_concepts = configs["hyperparameters"]["num_visual_concepts"]
        self.num_top_visual_concepts = configs["hyperparameters"][
            "num_top_visual_concepts"
        ]
        self.fine_to_coarse_pooling = configs["hyperparameters"][
            "fine_to_coarse_pooling"
        ]
        self.lambda_1 = configs["hyperparameters"]["lambda_1"]
        self.lambda_2 = configs["hyperparameters"]["lambda_2"]
        self.lambda_3 = configs["hyperparameters"]["lambda_3"]
        self.lambda_4 = configs["hyperparameters"]["lambda_4"]

        parameters = (
            list(self.encoder.parameters())
            + list(self.fine_level_classifier.parameters())
            + list(self.pseudo_label_classifier.parameters())
            + list(self.coarse_level_classifier.parameters())
        )
        self.optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)

        fine_to_coarse_mappings = json.load(
            open(configs["annotations"]["fine_to_coarse_mappings_path"])
        )
        self.fine_to_coarse_mappings = {
            int(k): v for k, v in fine_to_coarse_mappings.items()
        }
        self.coarse_to_fine_mappings = {}
        for f, c in self.fine_to_coarse_mappings.items():
            self.coarse_to_fine_mappings.setdefault(c, [])
            self.coarse_to_fine_mappings[c].append(f)

        self.best_mAP_avg = -float("inf")

    def _train_one_batch(self, batch_num, batch):
        self._start_training()
        loss = torch.zeros(1).to(self.device)

        (
            vnames,
            features,
            seq_lens,
            coarse_labels,
            fine_labels,
            coarse_weights,
            fine_weights,
        ) = batch
        features = features[:, : torch.max(seq_lens)].float().to(self.device)

        _, encoded_features = self.encoder(features)
        fine_cls_features, fine_cls_scores = self.fine_level_classifier(
            encoded_features
        )
        _, pseudo_scores = self.pseudo_label_classifier(encoded_features)
        classifier_weights = self.fine_level_classifier.out_layer.weight.detach()

        for i in range(len(seq_lens)):
            vname, seq_len = vnames[i], seq_lens[i]
            pseudo_label = self.pseudo_labels[vname]
            pseudo_label = torch.from_numpy(pseudo_label).long().to(self.device)

            loss += self._get_loss(
                f"{batch_num:04d}-{i}",
                vname,
                fine_cls_features[i, :seq_len],
                fine_cls_scores[i, :seq_len],
                fine_labels[i].to(self.device),
                fine_weights[i].to(self.device),
                pseudo_scores[i, :seq_len],
                pseudo_label,
                classifier_weights,
                coarse_labels[i].to(self.device),
                coarse_weights[i].to(self.device),
            )

        loss.backward()
        self.optimizer.step()

    def _get_loss(
        self,
        batch_num,
        vname,
        fine_cls_feature,
        fine_cls_score,
        fine_label,
        fine_weight,
        pseudo_score=None,
        pseudo_label=None,
        classifier_weights=None,
        coarse_label=None,
        coarse_weight=None,
    ):
        # L_mil
        loss_mil = self._get_loss_mil(fine_cls_score, fine_label, fine_weight)

        # L_pseudo
        loss_pseudo = self._get_loss_pseudo(pseudo_score, pseudo_label)

        # Hierarchy: clip -> visual concept -> fine-level action -> coarse-level action
        (
            visual_concepts,
            fine_level_reps,
            coarse_level_reps,
        ) = self._build_visual_concept_hierarchy(
            fine_cls_feature, pseudo_label, classifier_weights
        )

        # L_concept
        loss_concept = self._get_loss_concept(
            classifier_weights, fine_level_reps, fine_label
        )

        # L_coarse
        loss_coarse = self._get_loss_coarse(
            coarse_level_reps, coarse_label, coarse_weight
        )

        loss = (
            self.lambda_1 * loss_mil
            + self.lambda_2 * loss_pseudo
            + self.lambda_3 * loss_concept
            + self.lambda_4 * loss_coarse
        )

        if self.verbose:
            utils.log_message(
                f"batch {batch_num} loss={loss.item():.5f} l_mil={loss_mil.item():.5f} l_pseudo={loss_pseudo.item():.5f} l_concept={loss_concept.item():.5f} l_coarse={loss_coarse.item():.5f}"
            )

        return loss

    def _get_loss_mil(self, fine_cls_score, fine_label, fine_weight):
        fine_cls_score_agg = utils.get_aggregated_scores(fine_cls_score)
        return F.binary_cross_entropy_with_logits(
            fine_cls_score_agg, fine_label, weight=fine_weight
        )

    def _get_loss_pseudo(self, pseudo_score, pseudo_label):
        return F.cross_entropy(
            pseudo_score, pseudo_label, weight=self.pseudo_label_weights
        ) / len(pseudo_score)

    def _get_loss_concept(self, classifier_weights, fine_level_reps, fine_label):
        cosine_distances = 1 - F.cosine_similarity(
            classifier_weights, fine_level_reps, dim=1
        )
        return cosine_distances[fine_label.bool()].sum() / fine_label.sum()

    def _get_loss_coarse(self, coarse_level_reps, coarse_label, coarse_weight):
        _, coarse_score = self.coarse_level_classifier(coarse_level_reps)
        return F.binary_cross_entropy_with_logits(
            coarse_score.squeeze(1), coarse_label, weight=coarse_weight
        )

    def _build_visual_concept_hierarchy(
        self, fine_cls_feature, pseudo_label, classifier_weights
    ):
        visual_concepts = utils.get_visual_concepts_from_clip_features(
            fine_cls_feature, pseudo_label, self.num_visual_concepts, self.device
        )
        fine_level_reps = utils.get_fine_level_reps_from_visual_concepts(
            classifier_weights,
            visual_concepts,
            self.num_top_visual_concepts,
            self.device,
        )
        coarse_level_reps = utils.get_coarse_level_reps_from_fine_level_reps(
            fine_level_reps, self.coarse_to_fine_mappings, self.fine_to_coarse_pooling
        )
        return visual_concepts, fine_level_reps, coarse_level_reps

    def _train_one_epoch(self, epoch_num):
        utils.log_message("=" * 20)
        utils.log_message(f"Epoch {epoch_num}")
        utils.log_message("=" * 20)
        batch_num = 0
        for batch in self.loader:
            self._train_one_batch(batch_num, batch)
            batch_num += 1
        mAP_avg = self.evaluator.evaluate(epoch_num)
        if mAP_avg > self.best_mAP_avg:
            self.best_mAP_avg = mAP_avg
            self._save_models()

    def _generate_pseudo_labels(self):
        self.pseudo_labels, self.pseudo_label_weights = utils.get_pseudo_labels(
            self.loader,
            self.encoder,
            self.num_visual_concepts,
            self.clustering_sample_rate,
            self.device,
        )

    def _start_training(self):
        self.encoder.train()
        self.fine_level_classifier.train()
        self.pseudo_label_classifier.train()
        self.coarse_level_classifier.train()
        self.optimizer.zero_grad()

    def _save_models(self):
        torch.save(self.encoder.state_dict(), f"{self.output_dir}/encoder.pkl")
        torch.save(
            self.fine_level_classifier.state_dict(),
            f"{self.output_dir}/fine_level_classifier.pkl",
        )
        torch.save(
            self.pseudo_label_classifier.state_dict(),
            f"{self.output_dir}/pseudo_label_classifier.pkl",
        )
        torch.save(
            self.coarse_level_classifier.state_dict(),
            f"{self.output_dir}/coarse_level_classifier.pkl",
        )

    def train(self):
        for e in range(self.epochs):
            if e % 5 == 0:
                self._generate_pseudo_labels()
            self._train_one_epoch(e)
