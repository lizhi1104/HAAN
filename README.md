# Hierarchical Atomic Action Network
This repo contains the code for the paper:
Li, Z., He, L., & Xu, H. (2022). [Weakly-Supervised Temporal Action Detection for Fine-Grained Videos with Hierarchical Atomic Actions](https://arxiv.org/abs/2207.11805). arXiv preprint arXiv:2207.11805.


## Dependencies
The code is written and run with the following packages:
* Python 3.8.8
* PyTorch 1.7.1+cu110
* NumPy 1.20.1
* pandas 1.2.4
* scikit-learn 0.24.1


## Data
* `FineGym-new-split/` contains the new training/validation split we proposed for FineGym 99, with the same format as the original split on the [FineGym website](https://sdolivia.github.io/FineGym/).
* `dataset/` contains the annotations for each dataset. You can find the original annotations from the [FineGym](https://sdolivia.github.io/FineGym/) and [FineAction](https://deeperaction.github.io/datasets/fineaction.html) websites. We did some preprocessing and obtained the following files from the original annotations:
    * `fine_to_coarse_mappings.json`: a mapping from each fine-level class to its corresponding coarse-level class.
    * `video_names.npy`: the names of the videos.
    * `labels.npy`: the action labels within each video.
    * `segments.npy`: the action start/end timestamps within each video.
    * `sec2clip_ratios.npy`: the ratios to convert time in seconds to the corresponding clip index. For FineGym, we extract features with 24 fps and each clip has 16 consecutive frames, so the ratio is 24/16=1.5 for all videos. For FineAction, we use the features with a fixed 100 clips provided by the [FineAction competition page](https://codalab.lisn.upsaclay.fr/competitions/4386), so the ratios vary across videos.


## Instructions
### Data Preparation
Put the extracted I3D features under `dataset/FineAction/` and/or `dataset/FineGym` and update `features_path` in `config/fine_action.toml` and/or `config/fine_gym.toml` accordingly.
Features can be downloaded via [this link](https://drive.google.com/drive/folders/1IXh0k68j2m6bftQxgJMfkkhqRe0xyqnq?usp=sharing). We extracted FineGym features using [I3D](https://github.com/tomrunia/PyTorchConv3D), and FineAction features are from the [FineAction competition page](https://codalab.lisn.upsaclay.fr/competitions/4386). We use the `i3d_100` version of the features.
### Training
Run the following code, replacing `DATASET` with `FineAction` or `FineGym`, `EXP_NAME` with your experiment name, and `OUTPUT_DIR` with the directory where you want to store the results.
```
python main.py --dataset DATASET --exp-name EXP_NAME --output-dir OUTPUT_DIR
```
After the run finishes, four models `encoder.pkl, fine_level_classifier.pkl, pseudo_label_classifier.pkl, coarse_level_classifier.pkl` and one result file `results.csv` will be saved under `OUTPUT_DIR/DATASET/EXP_NAME`.
### Evaluation
Run the following code, replacing `DATASET` with `FineAction` or `FineGym`, and `INPUT_MODELS_DIR` with the directory where your models are stored.
```
python main.py --dataset DATASET --evaluation-only --input-models-dir INPUT_MODELS_DIR
```
Make sure to have `encoder.pkl` and `fine_level_classifier.pkl` under your `INPUT_MODELS_DIR`. The other two models `pseudo_label_classifier.pkl` and `coarse_level_classifier.pkl` are not needed for evaluation.
We also provide our pre-trained models under `output/FineAction/pre-trained` and `output/FineGym/pre-trained`.


## References
We referenced the following repos for the code:
* [ActivityNet](https://github.com/activitynet/ActivityNet)
* [wtalc-pytorch](https://github.com/sujoyp/wtalc-pytorch)


## Citation
Please cite the following work if you use this package.
```
@article{li2022weakly,
  title={Weakly-Supervised Temporal Action Detection for Fine-Grained Videos with Hierarchical Atomic Actions},
  author={Li, Zhi and He, Lu and Xu, Huijuan},
  journal={arXiv preprint arXiv:2207.11805},
  year={2022}
}
```


## Contact
If you have any questions, please contact the first author of the paper - Zhi Li (zhilicq@gmail.com).
