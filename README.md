<p align="center">
  <h1 align="center">MuDuo: Mutual Distillation with Dual Foundation Models for Semi-supervised PET/CT Organ Segmentation</h1>
</p>

<p align="center">
  <em>基于双基础模型互蒸馏的半监督 PET/CT 多器官分割框架 · MICCAI 2026</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/MICCAI-2026-blue" alt="MICCAI 2026"/>
  <img src="https://img.shields.io/badge/Python-3.10-green" alt="Python 3.10"/>
  <img src="https://img.shields.io/badge/PyTorch-2.6-red" alt="PyTorch 2.6"/>
  <img src="https://img.shields.io/badge/License-Apache%202.0-yellow" alt="License"/>
</p>

---

## 📌 Overview

**MuDuo** is a semi-supervised PET/CT organ segmentation framework that transfers complementary knowledge from two frozen foundation models into a lightweight student network. It uses **SAM-Med3D** to exploit CT anatomical priors and **SegAnyPET** to exploit PET functional localization, enabling robust pseudo-label generation under extremely limited annotation budgets.

Unlike conventional semi-supervised medical image segmentation methods that rely mainly on self-training or teacher-student consistency, MuDuo explicitly introduces cross-modal foundation-model guidance and consensus-based pseudo-label filtering for whole-body PET/CT multi-organ segmentation.

### ✨ Highlights

- 🔬 **Dual foundation-model distillation**  
  SAM-Med3D provides CT-driven anatomical guidance, while SegAnyPET provides PET-driven functional localization. Their complementary predictions are distilled into a compact 3D student model.

- 📊 **IoU-based consensus filtering**  
  An adaptive top-50% strategy keeps only high-consensus pseudo labels from the two expert branches, reducing noise propagation during semi-supervised training.

- 🎯 **Multi-point PET prompting**  
  A K=3 multi-point prompting strategy better captures heterogeneous tracer uptake patterns in PET volumes than single-point prompting.

- 🔄 **Dynamic pseudo-label fusion**  
  CT/PET expert predictions are randomly mixed to introduce perturbation and prevent the student model from simply memorizing its own predictions.

- 🏆 **Strong performance with very limited labels**  
  With only 5 labeled cases, MuDuo achieves 46.93% Dice and reduces HD95 by 41% compared with SemiSAM+ in our experimental setting.

---

## 🏗️ Framework

<p align="center">
  <img src="model.png" alt="MuDuo Overall Framework" width="100%"/>
</p>

<p align="center"><b>Figure 1.</b> Overview of MuDuo. The student model takes paired PET/CT volumes as input. For unlabeled data, two complementary teacher branches generate pseudo labels: the CT branch uses the student prediction as a mask prompt for SAM-Med3D, while the PET branch converts student features into point prompts for SegAnyPET. The two predictions are filtered by IoU consensus and fused to supervise the student through a consistency loss.</p>

### Qualitative Results

<p align="center">
  <img src="Fig2.png" alt="Qualitative Comparison" width="100%"/>
</p>

<p align="center"><b>Figure 2.</b> Qualitative comparison on PET/CT multi-organ segmentation. The upper row shows coronal segmentation results, and the lower row includes additional visualization with raw PET and the fully supervised reference.</p>

---

## 📁 Repository Structure

```text
MuDuo/
├── README.md
├── model.png / Fig2.png
│
├── code_MuDuo/                         # Core implementation of MuDuo
│   ├── DualSAM_train_MT_3D.py          # Main training script
│   ├── Dualsam_plus.py                 # SAM inference interface: mask / point prompt modes
│   ├── generate_pseudo_labels.py       # Offline pseudo-label generation
│   ├── semisam_plus.py                 # SemiSAM-style baseline inference
│   ├── val_3D.py                       # 3D validation: Dice / HD95
│   ├── test_3D.py / test_3D_util.py    # Testing utilities
│   ├── dataloaders/
│   │   └── autopet.py                  # AutoPET-Organ dataset loader and semi-supervised sampler
│   ├── networks/
│   │   ├── net_factory_3d.py           # 3D network factory
│   │   ├── unet_3D.py                  # 3D U-Net student backbone
│   │   └── vnet.py / nnunet.py         # Optional backbones
│   ├── segment_anything/
│   │   ├── build_sam3D.py              # SAM-Med3D / SegAnyPET builders
│   │   └── modeling/                   # 3D ViT encoder, prompt encoder, and mask decoder
│   └── utils/
│       ├── losses.py                   # Dice loss and related losses
│       └── ramps.py                    # Sigmoid ramp-up schedules
│
├── code/                               # Baseline implementations adapted from SSL4MIS
│   ├── train_mean_teacher_3D.py
│   ├── train_cross_pseudo_supervision_3D.py
│   ├── train_uncertainty_aware_mean_teacher_3D.py
│   ├── train_uncertainty_rectified_pyramid_consistency_3D.py
│   ├── train_fully_supervised_3D.py
│   └── ...
│
├── SAMMed3D/                           # SAM-Med3D utility repository
│   └── readme.md
│
└── SegAnyPET/                          # SegAnyPET utility repository
    └── README.md
```

---

## 🔧 Installation

### Requirements

- Python 3.10+
- CUDA 11.x / 12.x
- NVIDIA GPU with ≥24 GB memory  
  Recommended: A800 / A100 / H100 class GPUs for stable 3D training.

### Environment Setup

```bash
# 1. Clone this repository
git clone https://github.com/Wu-beining/MuDuo.git
cd MuDuo

# 2. Create a conda environment
conda create -n muduo python=3.10
conda activate muduo

# 3. Install PyTorch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# 4. Install dependencies
pip install numpy scipy scikit-image SimpleITK medpy
pip install tensorboardX tqdm prefetch_generator
pip install torchio opencv-python-headless matplotlib monai

# 5. Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Main Dependencies

| Package | Version | Purpose |
|---|---:|---|
| PyTorch | 2.6.0 | Deep learning framework |
| SimpleITK | latest | Medical image I/O |
| medpy | latest | Dice / HD95 evaluation |
| tensorboardX | latest | Training visualization |
| scipy | latest | Morphology and centroid computation |
| tqdm | latest | Progress bars |

---

## 📦 Dataset

### Dataset Provenance

This repository uses the **AutoPET-Organ** subset for PET/CT multi-organ segmentation.

AutoPET-Organ should be traced back to its original sources as follows:

1. **Original PET/CT images**  
   The original whole-body FDG-PET/CT images come from the AutoPET challenge / FDG-PET-CT-Lesions dataset. Users should obtain the original images from the official AutoPET website and follow the corresponding data access policy.

2. **Multi-organ annotations**  
   The 12-organ labels used by AutoPET-Organ were released by the SegAnyPET authors as `AutoPET-OrganlabelsTr.zip`. AutoPET-Organ contains 100 AutoPET cases with expert-examined annotations for 12 testing organs.

3. **Use in MuDuo**  
   MuDuo uses this AutoPET-Organ subset for semi-supervised PET/CT organ segmentation research. We do **not** claim ownership of the AutoPET images or the AutoPET-Organ annotations. Please cite and acknowledge the original AutoPET and SegAnyPET works when using this dataset.

We sincerely thank the **AutoPET** team for releasing the PET/CT imaging data and the **SegAnyPET** authors for providing valuable expert-examined organ annotations. Their efforts provide an important benchmark for the PET/CT multi-organ segmentation community and make this work possible.

### Organ Classes

AutoPET-Organ includes 12 organ classes:

| Index | Organ |
|---:|---|
| 1 | Liver |
| 2 | Left kidney |
| 3 | Right kidney |
| 4 | Heart |
| 5 | Spleen |
| 6 | Aorta |
| 7 | Prostate |
| 8 | Left lung lower lobe |
| 9 | Right lung lower lobe |
| 10 | Left lung upper lobe |
| 11 | Right lung upper lobe |
| 12 | Right lung middle lobe |

### Expected Directory Structure

```text
data/autopet/
├── images/
│   ├── {case_name}_0000.nii.gz      # CT image
│   └── {case_name}_0001.nii.gz      # PET image
├── labels_ok/
│   └── {case_name}.nii.gz           # AutoPET-Organ multi-organ label
├── lists_5/
│   ├── train.txt
│   └── val.txt
├── lists_10/
│   ├── train.txt
│   └── val.txt
└── lists_20/
    ├── train.txt
    └── val.txt
```

> **Data note.** This repository provides code and training scripts. Please download the original PET/CT images and the AutoPET-Organ labels from their official sources and ensure compliance with their licenses and data-use terms.

---

## 🔗 Pre-trained Weights

| Model | Role in MuDuo | Link |
|---|---|---|
| **SAM-Med3D Turbo** | CT teacher model | [Hugging Face](https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth) / [Google Drive](https://drive.google.com/file/d/1MuqYRQKIZb4YPtEraK8zTKKpp-dUQIR9/view?usp=sharing) |
| **SegAnyPET** | PET teacher model | [Hugging Face](https://huggingface.co/YichiZhang98/SegAnyPET) |
| **Prompt UNet** | PET prompt generator | Pre-train with the labeled split, as described below |

<details>
<summary>📋 Prompt UNet pre-training</summary>

The prompt generator is a PET-only supervised 3D U-Net. It should be pre-trained on the labeled subset:

```bash
cd code
python train_fully_supervised_3_pet_only.py \
    --root_path /data/autopet \
    --exp autopet_pet_fs_10 \
    --labeled_num 10 \
    --max_iterations 30000
```

After training, use `unet_3D_best_model.pth` as the `--ckpt_unet_prompt` checkpoint.

</details>

---

## 🚀 Quick Start

### Train MuDuo

```bash
cd code_MuDuo

python DualSAM_train_MT_3D.py \
    --root_path /data/autopet \
    --exp autopet/DualExpert_Top50 \
    --model unet_3D \
    --labeled_num 10 \
    --batch_size 4 \
    --labeled_bs 2 \
    --max_iterations 30000 \
    --base_lr 0.01 \
    --modality both \
    --num_points 3 \
    --ckpt_unet_prompt /path/to/prompt_unet_best.pth \
    --ckpt_sam_ct /path/to/sam_med3d_turbo.pth \
    --ckpt_sam_pet /path/to/seganypet_v1_or_v2.pth
```

The best model will be saved to:

```text
../model/{exp}/{model}/unet_3D_best_model.pth
```

<details>
<summary>📋 Main training arguments</summary>

| Argument | Default | Description |
|---|---:|---|
| `--root_path` | `/data/autopet` | Root directory of the AutoPET-Organ dataset |
| `--exp` | `autopet/DualExpert_Top50` | Experiment name and save path |
| `--model` | `unet_3D` | Student architecture |
| `--labeled_num` | 10 | Number of labeled training cases: 5 / 10 / 20 |
| `--batch_size` | 4 | Total batch size |
| `--labeled_bs` | 2 | Number of labeled cases per batch |
| `--max_iterations` | 30000 | Maximum training iterations |
| `--base_lr` | 0.01 | Initial learning rate with polynomial decay |
| `--modality` | `both` | Input modality: `ct`, `pet`, or `both` |
| `--num_points` | 3 | Number of PET prompt points |
| `--ema_decay` | 0.99 | EMA decay factor |
| `--patch_size` | `[128,128,128]` | Training patch size |

</details>

### Test / Inference

```bash
cd code_MuDuo

python test_3D.py \
    --root_path /data/autopet \
    --exp autopet/DualExpert_Top50 \
    --model unet_3D
```

### Baseline Training

```bash
cd code

# Mean Teacher
python train_mean_teacher_3D.py \
    --root_path /data/autopet \
    --labeled_num 10

# Cross Pseudo Supervision
python train_cross_pseudo_supervision_3D.py \
    --root_path /data/autopet \
    --labeled_num 10

# Uncertainty-Aware Mean Teacher
python train_uncertainty_aware_mean_teacher_3D.py \
    --root_path /data/autopet \
    --labeled_num 10

# Fully supervised baseline
python train_fully_supervised_3D.py \
    --root_path /data/autopet \
    --labeled_num 10
```

---

## 📊 Results

### Ablation Study with 5 Labeled Cases

<table>
  <thead>
    <tr>
      <th>Configuration</th>
      <th>Dice (%) ↑</th>
      <th>RAVD (%) ↓</th>
      <th>ASD ↓</th>
      <th>HD95 ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="5"><strong>(a) Teacher Configuration</strong></td>
    </tr>
    <tr>
      <td>SAM-Med3D only</td>
      <td>41.3 ± 22.3</td><td>38.6 ± 19.6</td><td>31.2 ± 18.3</td><td>14.8 ± 9.6</td>
    </tr>
    <tr>
      <td>SegAnyPET only</td>
      <td>38.9 ± 21.6</td><td>40.1 ± 20.3</td><td>34.7 ± 20.2</td><td>15.9 ± 10.2</td>
    </tr>
    <tr>
      <td><strong>Both teachers (MuDuo)</strong></td>
      <td><strong>46.9 ± 21.5</strong></td><td><strong>37.0 ± 18.8</strong></td><td><strong>23.7 ± 16.6</strong></td><td><strong>8.5 ± 7.3</strong></td>
    </tr>
    <tr>
      <td colspan="5"><strong>(b) IoU Filtering Strategy</strong></td>
    </tr>
    <tr>
      <td>w/o filtering</td>
      <td>44.6 ± 21.9</td><td>37.9 ± 18.9</td><td>26.5 ± 16.8</td><td>10.6 ± 8.3</td>
    </tr>
    <tr>
      <td>Fixed threshold, τ = 0.5</td>
      <td>45.8 ± 22.0</td><td>37.2 ± 18.6</td><td>25.2 ± 16.5</td><td>9.3 ± 7.9</td>
    </tr>
    <tr>
      <td><strong>Adaptive top-50% (MuDuo)</strong></td>
      <td><strong>46.9 ± 21.5</strong></td><td><strong>37.0 ± 18.8</strong></td><td><strong>23.7 ± 16.6</strong></td><td><strong>8.5 ± 7.3</strong></td>
    </tr>
    <tr>
      <td colspan="5"><strong>(c) Prompting Strategy</strong></td>
    </tr>
    <tr>
      <td>Single point, K = 1</td>
      <td>45.2 ± 22.0</td><td>37.6 ± 18.9</td><td>24.9 ± 16.2</td><td>9.1 ± 7.6</td>
    </tr>
    <tr>
      <td><strong>Multi-point, K = 3 (MuDuo)</strong></td>
      <td><strong>46.9 ± 21.5</strong></td><td><strong>37.0 ± 18.8</strong></td><td><strong>23.7 ± 16.6</strong></td><td><strong>8.5 ± 7.3</strong></td>
    </tr>
  </tbody>
</table>

---

## 📐 Technical Details

### Architecture

| Component | Description |
|---|---|
| Student network | 3D U-Net with two-channel PET/CT input |
| CT teacher | Frozen SAM-Med3D, used with mask prompts |
| PET teacher | Frozen SegAnyPET, used with point prompts |
| Prompt generator | Frozen PET-only 3D U-Net pre-trained on labeled data |

### Objective

The total training loss is:

$$
\mathcal{L}_{total} = \mathcal{L}_{sup} + \lambda(t)\mathcal{L}_{cons}.
$$

where:

- $\mathcal{L}_{sup} = 0.5 \times (\mathcal{L}_{CE} + \mathcal{L}_{Dice})$ is applied to labeled data.
- $\mathcal{L}_{cons}$ is the organ-wise Dice consistency loss applied to filtered pseudo labels from unlabeled data.
- $\lambda(t) = 0.1 \times \exp(-5(1 - t / 200)^2)$ is the sigmoid ramp-up consistency weight.

### Training Configuration

| Item | Setting |
|---|---|
| Optimizer | SGD, momentum = 0.9, weight decay = 1e-4 |
| LR schedule | Polynomial decay: lr = lr₀ × (1 − t / T)^0.9 |
| Initial LR | 0.01 |
| Maximum iterations | 30,000 |
| Patch size | 128 × 128 × 128 |
| Data augmentation | Random crop, random rotation, random flip |
| Inference | Sliding-window inference, stride = 64 |

### Evaluation Metrics

| Metric | Meaning |
|---|---|
| Dice (%) | Volumetric overlap; higher is better |
| HD95 | 95th percentile Hausdorff distance; lower is better |
| RAVD (%) | Relative absolute volume difference; lower is better |
| ASD | Average surface distance; lower is better |

---

## 🙏 Acknowledgements

This repository builds on several excellent open-source projects and datasets:

- [SSL4MIS](https://github.com/HiLab-git/SSL4MIS): semi-supervised medical image segmentation benchmark codebase.
- [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D): 3D medical segmentation foundation model.
- [SegAnyPET](https://github.com/YichiZhang98/SegAnyPET): universal promptable PET segmentation foundation model and the released AutoPET-Organ annotations.
- [Segment Anything](https://github.com/facebookresearch/segment-anything): the original Segment Anything Model.
- [AutoPET](https://autopet.grand-challenge.org): whole-body FDG-PET/CT imaging data.

We especially thank the SegAnyPET authors for releasing the expert-examined **AutoPET-Organ** labels. These annotations substantially support research on PET/CT multi-organ segmentation and provide an important public resource for the community.

---

## 📚 Citation

If you find this repository useful, please consider citing MuDuo:

```bibtex
@inproceedings{muduo2026,
  title     = {MuDuo: Mutual Distillation with Dual Foundation Models for Semi-supervised PET/CT Organ Segmentation},
  author    = {Author list},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI},
  year      = {2026}
}
```

If you use AutoPET-Organ or SegAnyPET-related resources, please also cite the corresponding SegAnyPET papers and the original AutoPET dataset paper:

```bibtex
@article{zhang2026developing,
  title   = {Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography},
  author  = {Zhang, Yichi and Xue, Le and Zhang, Wenbo and Li, Lanlan and Xiao, Feiyang and Liu, Yuchen and Zhang, Xiaohui and Zhang, Hongwei and Wang, Shuqi and Feng, Gang and Peng, Liling and Gao, Xin and Xu, Yuanfan and Qi, Yuan and Shi, Kuangyu and Zhang, Hong and Cheng, Yuan and Tian, Mei and Hu, Zixin},
  journal = {arXiv preprint arXiv:2603.11627},
  year    = {2026}
}

@inproceedings{zhang2025seganypet,
  title     = {SegAnyPET: Universal Promptable Segmentation from Positron Emission Tomography Images},
  author    = {Zhang, Yichi and Xue, Le and Zhang, Wenbo and Li, Lanlan and Liu, Yuchen and Jiang, Chen and Cheng, Yuan and Qi, Yuan},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month     = {October},
  year      = {2025},
  pages     = {21107--21116}
}

@article{gatidis2022autopet,
  title     = {A whole-body FDG-PET/CT dataset with manually annotated tumor lesions},
  author    = {Gatidis, Sergios and Hepp, Tobias and Fr{\"u}h, Marcel and La Foug{\`e}re, Christian and Nikolaou, Konstantin and Pfannenberg, Christina and Sch{\"o}lkopf, Bernhard and K{\"u}stner, Thomas and Cyran, Clemens and Rubin, Daniel},
  journal   = {Scientific Data},
  volume    = {9},
  number    = {1},
  pages     = {601},
  year      = {2022},
  publisher = {Nature Publishing Group UK London}
}
```

---

## 📧 Contact

For questions, please contact:

```text
2319723892@qq.com
```

