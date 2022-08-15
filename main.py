import os
import sys

import numpy as np
import torch

import options
from config.config import ConfigManager
from dataset.dataset import DatasetManager
from evaluation import Evaluator
from model import Model
from train import Trainer

if __name__ == "__main__":
    args = options.parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    configs = ConfigManager.get_configs(args.dataset)
    training_loader, test_loader = DatasetManager.get_dataset_loaders(configs)

    encoder = Model(
        configs["hyperparameters"]["feature_size"],
        configs["hyperparameters"]["encoded_feature_size"],
        configs["hyperparameters"]["encoded_feature_size"],
        configs["hyperparameters"]["dropout_rate"],
    ).to(device)
    fine_level_classifier = Model(
        configs["hyperparameters"]["encoded_feature_size"],
        configs["hyperparameters"]["encoded_feature_size"],
        configs["stats"]["num_fine_classes"],
        configs["hyperparameters"]["dropout_rate"],
    ).to(device)
    pseudo_label_classifier = Model(
        configs["hyperparameters"]["encoded_feature_size"],
        configs["hyperparameters"]["encoded_feature_size"],
        configs["hyperparameters"]["num_visual_concepts"],
        configs["hyperparameters"]["dropout_rate"],
    ).to(device)
    coarse_level_classifier = Model(
        configs["hyperparameters"]["encoded_feature_size"],
        configs["hyperparameters"]["encoded_feature_size"],
        1,
        configs["hyperparameters"]["dropout_rate"],
    ).to(device)

    if args.input_models_dir != "":
        encoder_path = os.path.join(args.input_models_dir, "encoder.pkl")
        fine_level_classifier_path = os.path.join(
            args.input_models_dir, "fine_level_classifier.pkl"
        )
        pseudo_label_classifier_path = os.path.join(
            args.input_models_dir, "pseudo_label_classifier.pkl"
        )
        coarse_level_classifier_path = os.path.join(
            args.input_models_dir, "coarse_level_classifier.pkl"
        )
        if os.path.exists(encoder_path):
            encoder.load_state_dict(torch.load(encoder_path))
        if os.path.exists(fine_level_classifier_path):
            fine_level_classifier.load_state_dict(
                torch.load(fine_level_classifier_path)
            )
        if os.path.exists(pseudo_label_classifier_path):
            pseudo_label_classifier.load_state_dict(
                torch.load(pseudo_label_classifier_path)
            )
        if os.path.exists(coarse_level_classifier_path):
            coarse_level_classifier.load_state_dict(
                torch.load(coarse_level_classifier_path)
            )

    output_dir = os.path.join(args.output_dir, args.dataset, args.exp_name)
    os.makedirs(output_dir, exist_ok=True)

    evaluator = Evaluator(
        configs, test_loader, encoder, fine_level_classifier, output_dir, device
    )

    if args.evaluation_only:
        evaluator.evaluate()
        sys.exit()

    trainer = Trainer(
        configs,
        training_loader,
        encoder,
        fine_level_classifier,
        pseudo_label_classifier,
        coarse_level_classifier,
        evaluator,
        output_dir,
        device,
    )

    trainer.train()
