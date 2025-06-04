# JiSAM (CVPR 2025)

Official code for CVPR 2025 paper [`[JiSAM: Alleviate Labeling Burden and Corner Case Problems in Autonomous Driving via Minimal Real-World Data]`](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_JiSAM_Alleviate_Labeling_Burden_and_Corner_Case_Problems_in_Autonomous_CVPR_2025_paper.pdf). Developed based on [`[OpenPCDet]`](https://github.com/open-mmlab/OpenPCDet)

## Highlights

* Jittering Augmentation and Memory-based Sectorized Alignment to bridge simulation-to-real gap (JiSAM).
* [JointTrainingDataset](./pcdet/datasets/joint_training_dataset/joint_training_dataset.py) to enable training with various datasets (any number you want), supporting future research on joint training with different LiDAR datasets.
* [Domain-aware Backbone](./pcdet/models/backbones_3d/spconv_backbone_joint_training.py) follows the joint dataset to add separate input kernels for different datasets.

## Getting Started

#### [Installation](./docs/INSTALL.md)

#### [Prepare Datasets](./docs/PREPARE_DATASETS.md)

#### Run example config

## ToDO List

- [ ] Upload simulation dataset from CARLA and complete dataset preparation instruction.
- [ ] Example training scripts
- [ ] Explore to use SOTA generative model to create simulation dataset.

## Citation 
If you find this project useful in your research, please consider cite:

```
@article{chen2025jisam,
  title={JiSAM: Alleviate Labeling Burden and Corner Case Problems in Autonomous Driving via Minimal Real-World Data},
  author={Chen, Runjian and Shao, Wenqi and Zhang, Bo and Shi, Shaoshuai and Jiang, Li and Luo, Ping},
  journal={arXiv preprint arXiv:2503.08422},
  year={2025}
}
```
