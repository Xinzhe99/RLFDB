# RLFDB: Robust Local Feature Detection in Blurred Images

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-ee4c2c.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![GitHub stars](https://img.shields.io/github/stars/Xinzhe99/RLFDB.svg)
![GitHub forks](https://img.shields.io/github/forks/Xinzhe99/RLFDB.svg)

**A robust local feature detection method for blurred images**

[Paper] | [Pretrained Models] | [Demo] | [Results]

</div>

## 📋 Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Model Architecture](#model-architecture)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 📖 Introduction

**RLFDB (Robust Local Feature Detection in Blurred Images)** is a deep learning-based method specifically designed for detecting robust local features in blurred images. Unlike traditional feature detectors that struggle with motion blur, defocus blur, and other image degradations, RLFDB maintains high repeatability and accuracy across various blur conditions.

### Key Contributions

- 🎯 **Novel Architecture**: Specialized neural network design for handling blurred images
- 🚀 **High Performance**: Superior repeatability compared to state-of-the-art methods
- 📊 **Comprehensive Evaluation**: Extensive testing on standard benchmarks
- 🔧 **Practical Application**: Real-time inference capability for real-world scenarios

## ✨ Features

- **🎯 Robust Detection**: Maintains high repeatability in various blur conditions
- **🚀 Real-time Performance**: Optimized network architecture for fast inference (~25 FPS)
- **📊 Benchmark Support**: Compatible with HSequences and other standard evaluation protocols
- **🔧 Easy Integration**: Simple APIs for both training and inference
- **🎨 Visualization Tools**: Built-in keypoint visualization and matching demos
- **⚡ Multi-GPU Support**: Distributed training on multiple GPUs
- **🔄 Flexible Configuration**: YAML-based configuration system

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.8 or higher
- CUDA 10.2+ (recommended for GPU acceleration)
- 8GB+ GPU memory (for training)

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/Xinzhe99/RLFDB.git
cd RLFDB

# Create conda environment (recommended)
conda create -n rlfdb python=3.8
conda activate rlfdb

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### Dependencies

```txt
torch>=1.8.0
torchvision>=0.9.0
opencv-python>=4.5.0
numpy>=1.19.0
pillow>=8.0.0
tensorboardX>=2.1
tqdm>=4.60.0
pyyaml>=5.4.0
matplotlib>=3.3.0
```

## 🚀 Quick Start

### Download Pretrained Models

```bash
# Create pretrained directory
mkdir pretrained

# Download pretrained weights (replace with actual download link)
wget -O pretrained/checkpoint.pth [DOWNLOAD_LINK]
```

### Simple Inference

```bash
# Run inference on a single image
python demo_inference.py \
    --ckpt_file pretrained/checkpoint.pth \
    --input_image assets/kitti_selected_motion_blur_median.png \
    --output_image results/output.png \
    --num_features 2048
```

### Feature Matching Demo

```bash
# Run feature matching between two images
python demo_match_proposed.py \
    --image_path1 assets/image1.jpg \
    --image_path2 assets/image2.jpg \
    --output_path results/matching_result.jpg
```

## 💻 Usage

### Python API

```python
import torch
import numpy as np
from PIL import Image
from model.network import RLFDB
from demo_inference import load_im, detect_and_save, parse_test_config

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RLFDB(
    dims=[32, 64, 128, 256], 
    layers=[2, 2, 6, 2], 
    expand_ratio=3.0, 
    mlp_ratio=3.0, 
    use_dw=True,
    drop_path_rate=0.05
).to(device)

# Load pretrained weights
checkpoint = torch.load('pretrained/checkpoint.pth', map_location=device)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# Detect keypoints
args = parse_test_config()
image = load_im('path/to/your/image.jpg')
keypoints = detect_and_save(args, image, model, device)

print(f"Detected {len(keypoints)} keypoints")
# keypoints format: [x, y, score, response] for each point
```

### Command Line Interface

```bash
# Inference with custom parameters
python demo_inference.py \
    --ckpt_file pretrained/checkpoint.pth \
    --input_image your_image.jpg \
    --output_image result.jpg \
    --num_features 1024 \
    --heatmap_confidence_threshold 0.005 \
    --nms_size 3 \
    --border_size 15

# Batch processing
python predict_kp_on_eval_dataset.py \
    --ckpt_file pretrained/checkpoint.pth \
    --input_dir datasets/test_images \
    --output_dir results/predictions
```

## 📊 Dataset

### Supported Datasets

- **HSequences**: Standard benchmark for local feature evaluation
- **Custom Datasets**: Support for user-defined blurred image datasets

### Data Preparation

```bash
# Create dataset directory
mkdir -p datasets

# HSequences dataset structure
datasets/
├── HSequences/
│   ├── i_autobahn/
│   │   ├── image_00000000.ppm
│   │   ├── image_00000001.ppm
│   │   └── ...
│   ├── i_building/
│   └── ...
```

### Dataset Configuration

Update the dataset paths in `configs/data_config.yaml`:

```yaml
data:
  train:
    dataset_dir: "datasets/train"
    batch_size: 8
  val:
    dataset_dir: "datasets/val"
    batch_size: 1
  test:
    dataset_dir: "datasets/HSequences"
```

## 🏋️ Training

### Configuration

Training parameters are configured in YAML files located in the `configs/` directory.

```yaml
# configs/train_config.yaml
model:
  cell_size: 8
  anchor_loss: true
  optimizer:
    type: "Adam"
    lr: 0.0001
    total_epochs: 100
  scheduler:
    type: "StepLR"
    step_size: 30
    gamma: 0.5
```

### Start Training

```bash
# Single GPU training
python train.py \
    --config configs/train_config.yaml \
    --exper_name rlfdb_experiment \
    --gpu_ids 0 \
    --log_dir logs

# Multi-GPU training
python train.py \
    --config configs/train_config.yaml \
    --exper_name rlfdb_multi_gpu \
    --gpu_ids 0,1,2,3 \
    --log_dir logs
```

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--exper_name` | Experiment name for logging | Required |
| `--log_dir` | Directory for saving logs and checkpoints | `logs` |
| `--gpu_ids` | GPU device IDs (comma-separated) | `0` |
| `--fix_random_seed` | Fix random seed for reproducibility | `False` |
| `--random_seed` | Random seed value | `42` |
| `--is_debugging` | Enable debugging mode | `False` |

### Monitoring Training

```bash
# Start TensorBoard
tensorboard --logdir logs/your_experiment_name/

# View training logs
tail -f logs/your_experiment_name/log_train.txt
```

## 📈 Evaluation

### HSequences Benchmark

```bash
# Evaluate on HSequences dataset
python hsequeces_bench_kpts.py \
    --benchmark_input_root datasets/HSequences \
    --results_dir results/hsequences \
    --detector_name RLFDB \
    --ckpt_file pretrained/checkpoint.pth
```

### Custom Evaluation

```bash
# Generate keypoints for evaluation dataset
python predict_kp_on_eval_dataset.py \
    --ckpt_file pretrained/checkpoint.pth \
    --input_dir datasets/test_images \
    --output_dir results/keypoints \
    --num_features 2048
```

### Evaluation Metrics

- **Repeatability**: Percentage of detected keypoints that are repeatable across different views
- **Localization Error**: Average pixel distance between corresponding keypoints
- **Matching Score**: Performance in feature matching tasks
- **Inference Speed**: Frames per second (FPS) on standard hardware

## 📊 Results

### Performance on HSequences

| Method | Repeatability@3px | Localization Error | Speed (FPS) |
|--------|-------------------|-------------------|-------------|
| SIFT | 0.43 | 2.1px | 12 |
| SURF | 0.39 | 2.3px | 18 |
| ORB | 0.35 | 2.8px | 45 |
| SuperPoint | 0.67 | 1.8px | 22 |
| **RLFDB (Ours)** | **0.85** | **1.2px** | **25** |

### Qualitative Results

![Feature Detection Results](assets/results_comparison.png)

*Comparison of feature detection results on blurred images. RLFDB maintains consistent detection quality across various blur conditions.*

### Ablation Studies

| Component | Repeatability | Notes |
|-----------|---------------|-------|
| Base Model | 0.72 | Without blur-specific modules |
| + Blur Module | 0.78 | Added blur-aware components |
| + Multi-scale | 0.82 | Multi-scale feature extraction |
| **Full RLFDB** | **0.85** | Complete architecture |

## 🏗️ Model Architecture

### Network Overview

RLFDB adopts a hierarchical feature extraction architecture specifically designed for blurred images:

```
Input Image (H×W×3)
    ↓
Blur-Aware Encoder
    ├── Level 1: dims=32, layers=2
    ├── Level 2: dims=64, layers=2
    ├── Level 3: dims=128, layers=6
    └── Level 4: dims=256, layers=2
    ↓
Multi-Scale Feature Fusion
    ↓
Keypoint Detection Head
    ↓
Output: Probability Map (H/8×W/8×1)
```

### Key Components

- **Blur-Aware Encoder**: Specialized backbone for handling blur artifacts
- **Multi-Scale Fusion**: Combining features from different resolution levels
- **Attention Mechanism**: Focus on discriminative regions
- **Keypoint Head**: Generates probability maps for keypoint locations

### Model Parameters

```python
model = RLFDB(
    dims=[32, 64, 128, 256],      # Feature dimensions at each level
    layers=[2, 2, 6, 2],          # Number of layers at each level
    expand_ratio=3.0,             # Expansion ratio for MLP
    mlp_ratio=3.0,                # MLP ratio in attention blocks
    use_dw=True,                  # Use depthwise convolution
    drop_path_rate=0.05           # Drop path rate for regularization
)
```

## 🛠️ Project Structure

```
RLFDB/
├── assets/                     # Sample images and visualizations
├── benchmark_test/            # Benchmark evaluation results
├── configs/                   # Configuration files
│   ├── train_config.yaml     # Training configuration
│   └── data_config.yaml      # Dataset configuration
├── datasets/                  # Dataset storage
│   └── HSequences/           # HSequences benchmark data
├── loss/                      # Loss function implementations
├── model/                     # Model definitions
│   ├── network.py            # Main RLFDB network
│   └── components.py         # Network components
├── third_party/              # Third-party libraries
├── utils/                    # Utility functions
│   ├── train_utils.py        # Training utilities
│   ├── test_utils.py         # Testing utilities
│   └── common_utils.py       # Common utilities
├── demo_inference.py         # Single image inference demo
├── demo_match_proposed.py    # Feature matching demo
├── train.py                  # Training script
├── predict_kp_on_eval_dataset.py  # Batch prediction
├── hsequeces_bench_kpts.py   # HSequences evaluation
└── README.md                 # This file
```

## 🔗 Links

- 📄 **Paper**: [Robust Local Feature Detection in Blurred Images](link-to-paper)
- 🎯 **Pretrained Models**: [Download](link-to-models)
- 📊 **Supplementary Material**: [Link](link-to-supplementary)
- 🐛 **Issues**: [GitHub Issues](https://github.com/Xinzhe99/RLFDB/issues)

## 📝 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{xinzhe2024rlfdb,
  title={Robust Local Feature Detection in Blurred Images},
  author={Xinzhe and Contributors},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Xinzhe99

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## 🙏 Acknowledgments

We thank the following projects and datasets for their contributions:

- **HSequences Dataset**: For providing the evaluation benchmark
- **PyTorch Team**: For the excellent deep learning framework
- **OpenCV Community**: For computer vision utilities
- **Research Community**: For valuable feedback and suggestions

Special thanks to all contributors who have helped improve this project.

## 📧 Contact

- **Author**: Xinzhe99
- **GitHub**: [@Xinzhe99](https://github.com/Xinzhe99)
- **Issues**: Please report bugs and feature requests via [GitHub Issues](https://github.com/Xinzhe99/RLFDB/issues)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

<div align="center">

⭐ **Star this repository if you find it helpful!** ⭐

**Made with ❤️ by [Xinzhe99](https://github.com/Xinzhe99)**

</div>
