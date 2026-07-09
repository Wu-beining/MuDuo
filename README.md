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

## 📌 项目简介

**MuDuo** 是一个面向 **PET/CT 多器官分割** 的半监督学习框架。它利用两个互补的医学基础模型作为教师模型：

- **SAM-Med3D**：侧重 CT 结构影像中的解剖先验；
- **SegAnyPET**：侧重 PET 功能影像中的代谢/摄取定位信息。

MuDuo 的核心思想是：在只有少量人工标注的情况下，通过 CT 与 PET 两条教师分支生成互补伪标签，并利用一致性筛选机制过滤低质量伪标签，从而更稳定地训练轻量级 3D 学生网络。

该框架主要面向以下问题：

1. PET/CT 多器官标注成本高，难以大规模获取精细标签；
2. PET 图像边界模糊、解剖结构弱，单独依赖 PET 容易产生定位偏差；
3. CT 与 PET 信息互补，但直接融合容易受到伪标签噪声影响；
4. 现有半监督方法大多依赖单模型自训练，难以充分利用医学基础模型的先验能力。

---

## ✨ 方法亮点

- 🔬 **双基础模型互补蒸馏**  
  使用 SAM-Med3D 提供 CT 解剖先验，使用 SegAnyPET 提供 PET 功能定位先验，将两类基础模型的知识共同蒸馏到轻量级学生模型中。

- 📊 **IoU 共识过滤机制**  
  对 CT 教师分支与 PET 教师分支产生的伪标签进行一致性评估，采用自适应 Top-50% 筛选策略，仅保留高共识伪标签，降低噪声传播风险。

- 🎯 **多点 PET Prompt 策略**  
  采用 K=3 的多点提示采样方式，更好地覆盖 PET 中常见的异质性摄取区域，缓解单点 prompt 对复杂器官定位不足的问题。

- 🔄 **动态伪标签融合**  
  在训练过程中动态融合 CT/PET 教师预测，引入适度扰动，避免学生模型简单记忆自身预测。

- 🏆 **低标注量场景下表现稳定**  
  在仅使用极少量标注样本的设置下，MuDuo 仍能获得较好的 Dice 与 HD95 表现，体现出较强的标注效率。

---

## 🏗️ 方法框架

<p align="center">
  <img src="model.png" alt="MuDuo Overall Framework" width="100%"/>
</p>

<p align="center"><b>图 1.</b> MuDuo 整体框架。学生模型接收配准后的 PET/CT 双模态体数据作为输入。对于无标签样本，CT 分支将学生预测作为 mask prompt 输入 SAM-Med3D，PET 分支将学生特征转换为 point prompt 输入 SegAnyPET。两条分支的预测经过 IoU 共识筛选后融合为伪标签，并通过一致性损失监督学生模型。</p>

### 定性结果

<p align="center">
  <img src="Fig2.png" alt="Qualitative Comparison" width="100%"/>
</p>

<p align="center"><b>图 2.</b> PET/CT 多器官分割定性对比。上排展示冠状位分割结果，下排展示原始 PET、全监督基线及不同方法的可视化结果。</p>

---

## 📁 项目结构

```text
MuDuo/
├── README.md
├── model.png / Fig2.png                # 论文配图
│
├── code_MuDuo/                         # MuDuo 核心代码
│   ├── DualSAM_train_MT_3D.py          # 主训练脚本：双基础模型互蒸馏
│   ├── Dualsam_plus.py                 # SAM 推理接口：mask / point prompt
│   ├── generate_pseudo_labels.py       # 独立伪标签生成脚本
│   ├── semisam_plus.py                 # SemiSAM 风格基线推理
│   ├── val_3D.py                       # 3D 验证：Dice / HD95
│   ├── test_3D.py / test_3D_util.py    # 测试与推理工具
│   ├── dataloaders/
│   │   └── autopet.py                  # AutoPET-Organ 数据加载器与半监督采样器
│   ├── networks/
│   │   ├── net_factory_3d.py           # 3D 网络工厂
│   │   ├── unet_3D.py                  # 3D U-Net 学生模型
│   │   └── vnet.py / nnunet.py         # 其他可选骨干网络
│   ├── segment_anything/               # 3D SAM 类模型实现
│   │   ├── build_sam3D.py              # SAM-Med3D / SegAnyPET 构建入口
│   │   └── modeling/                   # ViT 3D 编码器、Prompt 编码器、Mask 解码器
│   └── utils/
│       ├── losses.py                   # Dice Loss 等损失函数
│       └── ramps.py                    # Sigmoid ramp-up 权重调度
│
├── code/                               # 对比实验代码，基于 SSL4MIS 修改
│   ├── train_mean_teacher_3D.py        # Mean Teacher
│   ├── train_cross_pseudo_supervision_3D.py  # CPS
│   ├── train_uncertainty_aware_mean_teacher_3D.py  # UA-MT
│   ├── train_uncertainty_rectified_pyramid_consistency_3D.py  # URPC
│   ├── train_fully_supervised_3D.py    # 全监督基线
│   └── ...
│
├── SAMMed3D/                           # SAM-Med3D 相关工具与说明
│   └── readme.md
│
└── SegAnyPET/                          # SegAnyPET 相关工具与说明
    └── README.md
```

---

## 🔧 环境安装

### 基础环境

- Python 3.10+
- CUDA 11.x / 12.x
- NVIDIA GPU，建议显存 ≥ 24GB；大规模实验推荐 A800 / A100 80GB

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/Wu-beining/MuDuo.git
cd MuDuo

# 2. 创建 Conda 环境
conda create -n muduo python=3.10
conda activate muduo

# 3. 安装 PyTorch，请根据本机 CUDA 版本选择对应命令
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# 4. 安装常用依赖
pip install numpy scipy scikit-image SimpleITK medpy
pip install tensorboardX tqdm prefetch_generator
pip install torchio opencv-python-headless matplotlib monai

# 5. 检查环境
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 关键依赖说明

| 依赖库 | 建议版本 | 主要用途 |
|---|---:|---|
| PyTorch | 2.6.0 | 深度学习框架 |
| SimpleITK | latest | 医学图像 NIfTI 读写 |
| medpy | latest | Dice / HD95 等医学分割指标 |
| tensorboardX | latest | 训练日志与可视化 |
| scipy | latest | 形态学操作、重心计算等 |
| tqdm | latest | 训练进度显示 |

---

## 📦 数据集说明

### AutoPET-Organ 子集来源

本项目使用的 **AutoPET-Organ** 是基于 AutoPET PET/CT 数据构建的多器官标注子集。为避免数据来源不清，本文档对其来源说明如下：

| 项目 | 说明 |
|---|---|
| 原始 PET/CT 图像 | 来自 AutoPET 官方数据集。请按照 AutoPET 官方网站的要求申请和下载原始图像。 |
| 多器官标注 | 来自 SegAnyPET 团队发布的 `AutoPET-OrganlabelsTr.zip`。 |
| 子集规模 | 100 个 AutoPET case。 |
| 标注类别 | 12 类器官，包括 liver、left kidney、right kidney、heart、spleen、aorta、prostate、left lung lower lobe、right lung lower lobe、left lung upper lobe、right lung upper lobe、right lung middle lobe。 |
| 本项目用途 | 用于评估半监督 PET/CT 多器官分割方法在低标注量场景下的表现。 |

> **重要声明**  
> MuDuo 不重新发布 AutoPET 原始图像，也不声明拥有 AutoPET-Organ 标注的所有权。使用者应分别遵循 AutoPET 官方数据协议以及 SegAnyPET / AutoPET-Organ 标注发布方的使用要求。若你使用该数据子集开展研究，请同时引用 AutoPET 与 SegAnyPET / AutoPET-Organ 的相关工作。

我们特别感谢 SegAnyPET 团队整理并公开 AutoPET-Organ 多器官标注。该标注子集将原本主要面向肿瘤病灶分割的 AutoPET PET/CT 数据进一步扩展到多器官分割任务，为 PET/CT 多器官分割、基础模型泛化评估以及半监督学习研究提供了非常有价值的公共资源。

### 推荐数据目录结构

```text
data/autopet/
├── images/
│   ├── {case_name}_0000.nii.gz      # CT 图像
│   └── {case_name}_0001.nii.gz      # PET 图像
├── labels_ok/
│   └── {case_name}.nii.gz           # AutoPET-Organ 多器官标注
├── lists_5/
│   ├── train.txt                    # 5 个标注样本设置
│   └── val.txt
├── lists_10/
│   ├── train.txt                    # 10 个标注样本设置
│   └── val.txt
└── lists_20/
    ├── train.txt                    # 20 个标注样本设置
    └── val.txt
```

---

## 🔗 预训练权重

| 模型 | 用途 | 下载链接 |
|---|---|---|
| SAM-Med3D Turbo | CT 教师模型 | [HuggingFace](https://huggingface.co/blueyo0/SAM-Med3D/blob/main/sam_med3d_turbo.pth) / [Google Drive](https://drive.google.com/file/d/1MuqYRQKIZb4YPtEraK8zTKKpp-dUQIR9/view?usp=sharing) |
| SegAnyPET v1 | PET 教师模型 | [HuggingFace](https://huggingface.co/YichiZhang98/SegAnyPET) |
| Prompt UNet | PET prompt 生成器 | 需要在标注训练集上自行预训练 |

<details>
<summary>📋 Prompt UNet 预训练方法</summary>

Prompt Generator 是一个 PET-only 的全监督 3D U-Net。它需要先在可用标注数据上训练，然后在 MuDuo 中冻结，用于生成 PET 分支的 point prompt。

```bash
cd code
python train_fully_supervised_3_pet_only.py \
    --root_path /data/autopet \
    --exp autopet_pet_fs_10 \
    --labeled_num 10 \
    --max_iterations 30000
```

训练完成后，将得到的 `unet_3D_best_model.pth` 路径传入 MuDuo 训练脚本中的 `--ckpt_unet_prompt` 参数。

</details>

---

## 🚀 快速开始

### 训练 MuDuo

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
    --ckpt_sam_pet /path/to/seganypet_v1.pth
```

训练完成后，最优模型默认保存为：

```text
../model/{exp}/{model}/unet_3D_best_model.pth
```

<details>
<summary>📋 常用训练参数说明</summary>

| 参数 | 默认/示例 | 说明 |
|---|---:|---|
| `--root_path` | `/data/autopet` | AutoPET-Organ 数据根目录 |
| `--exp` | `autopet/DualExpert_Top50` | 实验名称，决定模型保存路径 |
| `--model` | `unet_3D` | 学生模型结构 |
| `--labeled_num` | `10` | 标注样本数量，可设为 5 / 10 / 20 |
| `--batch_size` | `4` | 总 batch size |
| `--labeled_bs` | `2` | 每个 batch 中的标注样本数量 |
| `--max_iterations` | `30000` | 最大训练迭代数 |
| `--base_lr` | `0.01` | 初始学习率 |
| `--modality` | `both` | 输入模态，可选 `ct` / `pet` / `both` |
| `--num_points` | `3` | PET 分支 point prompt 数量 |
| `--ema_decay` | `0.99` | EMA teacher 衰减率 |
| `--patch_size` | `[128,128,128]` | 训练 patch 大小 |

</details>

### 测试与推理

```bash
cd code_MuDuo

python test_3D.py \
    --root_path /data/autopet \
    --exp autopet/DualExpert_Top50 \
    --model unet_3D
```

### 运行对比实验

```bash
cd code

# Mean Teacher
python train_mean_teacher_3D.py \
    --root_path /data/autopet \
    --labeled_num 10

# Cross Pseudo Supervision, CPS
python train_cross_pseudo_supervision_3D.py \
    --root_path /data/autopet \
    --labeled_num 10

# Uncertainty-Aware Mean Teacher, UA-MT
python train_uncertainty_aware_mean_teacher_3D.py \
    --root_path /data/autopet \
    --labeled_num 10

# Fully Supervised baseline
python train_fully_supervised_3D.py \
    --root_path /data/autopet \
    --labeled_num 10
```

---

## 📊 实验结果

### 消融实验：5 个标注样本设置

<table>
  <thead>
    <tr>
      <th>Configuration</th>
      <th>Dice(%)↑</th>
      <th>RAVD(%)↓</th>
      <th>ASD↓</th>
      <th>HD95↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="5"><strong>(a) Teacher Configuration</strong></td>
    </tr>
    <tr>
      <td>SAM-Med3D only</td>
      <td>41.3±22.3</td><td>38.6±19.6</td><td>31.2±18.3</td><td>14.8±9.6</td>
    </tr>
    <tr>
      <td>SegAnyPET only</td>
      <td>38.9±21.6</td><td>40.1±20.3</td><td>34.7±20.2</td><td>15.9±10.2</td>
    </tr>
    <tr>
      <td><strong>Both, Ours</strong></td>
      <td><strong>46.9±21.5</strong></td><td><strong>37.0±18.8</strong></td><td><strong>23.7±16.6</strong></td><td><strong>8.5±7.3</strong></td>
    </tr>
    <tr>
      <td colspan="5"><strong>(b) IoU Filtering Strategies</strong></td>
    </tr>
    <tr>
      <td>w/o filtering</td>
      <td>44.6±21.9</td><td>37.9±18.9</td><td>26.5±16.8</td><td>10.6±8.3</td>
    </tr>
    <tr>
      <td>Fixed threshold, τ=0.5</td>
      <td>45.8±22.0</td><td>37.2±18.6</td><td>25.2±16.5</td><td>9.3±7.9</td>
    </tr>
    <tr>
      <td><strong>Adaptive top-50%, Ours</strong></td>
      <td><strong>46.9±21.5</strong></td><td><strong>37.0±18.8</strong></td><td><strong>23.7±16.6</strong></td><td><strong>8.5±7.3</strong></td>
    </tr>
    <tr>
      <td colspan="5"><strong>(c) Prompting Strategies</strong></td>
    </tr>
    <tr>
      <td>Single-point, K=1</td>
      <td>45.2±22.0</td><td>37.6±18.9</td><td>24.9±16.2</td><td>9.1±7.6</td>
    </tr>
    <tr>
      <td><strong>Multi-point, K=3, Ours</strong></td>
      <td><strong>46.9±21.5</strong></td><td><strong>37.0±18.8</strong></td><td><strong>23.7±16.6</strong></td><td><strong>8.5±7.3</strong></td>
    </tr>
  </tbody>
</table>

---

## 📐 技术细节

### 模型配置

| 组件 | 设置 |
|---|---|
| 学生网络 | 3D U-Net，输入通道为 PET + CT |
| CT 教师 | SAM-Med3D，冻结参数 |
| PET 教师 | SegAnyPET，冻结参数 |
| Prompt 生成器 | PET-only 3D U-Net，预训练后冻结 |
| 输入 patch | 128 × 128 × 128 |
| 推理方式 | Sliding-window inference，stride=64 |

### 损失函数

MuDuo 的总损失由有监督分割损失和无监督一致性损失组成：

$$
\mathcal{L}_{total} = \mathcal{L}_{sup} + \lambda(t) \cdot \mathcal{L}_{cons}
$$

其中：

- $\mathcal{L}_{sup}$：标注样本上的监督损失，采用 Cross-Entropy 与 Dice Loss 的组合；
- $\mathcal{L}_{cons}$：无标签样本上的一致性损失，仅对通过 IoU 共识筛选的伪标签计算；
- $\lambda(t)$：Sigmoid ramp-up 权重，用于逐步增强无监督损失的影响。

### 训练设置

| 配置项 | 设置 |
|---|---|
| 优化器 | SGD |
| Momentum | 0.9 |
| Weight decay | 1e-4 |
| 学习率策略 | Poly learning-rate schedule |
| 初始学习率 | 0.01 |
| 最大迭代数 | 30,000 |
| 数据增强 | 随机裁剪、随机旋转、随机翻转 |

### 评估指标

| 指标 | 含义 | 趋势 |
|---|---|---|
| Dice (%) | 体素级重叠程度 | 越高越好 |
| HD95 | 95% Hausdorff Distance | 越低越好 |
| RAVD (%) | Relative Absolute Volume Difference | 越低越好 |
| ASD | Average Surface Distance | 越低越好 |

---

## 📚 Citation

如果你认为本项目对你的研究有帮助，请考虑引用 MuDuo 以及相关基础模型/数据集工作。

```bibtex
@inproceedings{muduo2026,
  title     = {MuDuo: Mutual Distillation with Dual Foundation Models for Semi-supervised PET/CT Organ Segmentation},
  author    = {Anonymous},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI},
  year      = {2026}
}
```

同时，请根据实际使用情况引用以下资源：

- AutoPET / FDG-PET/CT 数据集；
- SegAnyPET 及其 AutoPET-Organ 标注；
- SAM-Med3D；
- SSL4MIS；
- Segment Anything。

---

## 🙏 致谢

本项目的实现受益于多个开源项目、公开数据集与医学影像基础模型。我们真诚感谢以下工作及其作者：

- [SSL4MIS](https://github.com/HiLab-git/SSL4MIS)：提供了半监督医学图像分割基准代码框架；
- [SAM-Med3D](https://github.com/uni-medical/SAM-Med3D)：提供了通用 3D 医学图像分割基础模型；
- [SegAnyPET](https://github.com/YichiZhang98/SegAnyPET)：提供了面向 PET 图像的通用可提示分割模型，并发布了 AutoPET-Organ 多器官标注；
- [Segment Anything](https://github.com/facebookresearch/segment-anything)：为 promptable segmentation 提供了基础方法框架；
- [AutoPET](https://autopet.grand-challenge.org)：提供了大规模全身 FDG-PET/CT 数据资源。

我们尤其感谢 **SegAnyPET 团队** 对 AutoPET 数据中的 100 个 case 进行多器官标注整理，并公开 `AutoPET-OrganlabelsTr.zip`。这些高质量标注为 PET/CT 多器官分割研究提供了宝贵资源，也推动了 PET 基础模型、跨模态分割和低标注量医学图像分割的发展。

---

## 📧 联系方式

如果你在使用过程中遇到问题，欢迎通过以下邮箱联系：

```text
2319723892@qq.com
```

---

## ⚠️ 免责声明

本仓库仅用于学术研究。使用者需自行确认相关数据集、预训练权重和第三方代码的许可协议。对于 AutoPET 原始图像与 AutoPET-Organ 标注，请遵循其各自发布方的使用条款和引用要求。
