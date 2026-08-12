# Semantic-Style-Transfer

Official implementation of my master's thesis project on **semantic-aware arbitrary image style transfer**.

This project aims to improve arbitrary style transfer by introducing semantic information into the feature correspondence process and by selecting semantically appropriate style images for the given content image.

The framework combines **S2WAT**, **AdaFormer**, and **DINOv2-based semantic similarity** for style transfer. In addition, **OWL-ViT** and **CLIP** are used to perform semantic-aware style image matching before inference.

## Overview

Conventional arbitrary style transfer methods mainly establish content-style correspondences based on feature statistics or attention mechanisms. However, without explicit semantic constraints, style features from unrelated regions may be incorrectly transferred, resulting in unreasonable color or texture mappings.

This project introduces semantic information into the style transfer process to improve the correspondence between content and style features. It also includes a style matching mechanism that automatically selects a semantically suitable style image from a candidate style set.

The overall pipeline consists of two main components:

1. **Semantic-aware Style Transfer**

   * S2WAT-based visual feature encoder
   * DINOv2 semantic feature extraction
   * Semantic similarity-guided AdaFormer
   * Multi-level feature decoding

2. **Reasonable Style Matching**

   * OWL-ViT object detection
   * Semantic group matching
   * CLIP similarity evaluation
   * Automatic style image selection

## Project Structure

```text
Semantic-Style-Transfer/
├── README.md
├── LICENSE
├── requirements.txt
├── train.py
├── test.py
├── net.py
├── data_preprocess.py
├── dataset_sampler.py
├── object_detection.py
├── scheduler.py
├── tools.py
│
├── model/
│   ├── __init__.py
│   ├── AdaFormer.py
│   ├── configuration.py
│   ├── s2wat.py
│   ├── transformer_components.py
│   └── transformer_tools.py
│
├── input/
│   ├── Train/
│   └── Test/
│
├── pre_trained_models/
│   ├── checkpoint/
│   └── vgg_normalised.pth
│
└── output/
```

The `input/`, `output/`, and pretrained weight files are excluded from Git tracking and should be prepared locally.

## Environment

The project was developed using Python 3.12.

The development environment used:

```text
Python       3.12
PyTorch      2.10.0+cu130
Torchvision  0.25.0+cu130
NumPy        2.4.1
Pillow       12.0.0
tqdm         4.67.1
Transformers 5.3.0
```

A CUDA-enabled GPU is recommended for training and inference.

### Installation

Clone the repository:

```bash
git clone https://github.com/R-Hsiao/Semantic-Style-Transfer.git
cd Semantic-Style-Transfer
```

Create a Conda environment:

```bash
conda create -n S2WAT python=3.12
conda activate S2WAT
```

Install a compatible version of PyTorch and Torchvision for your CUDA environment first.

Then install the remaining dependencies:

```bash
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

The current `requirements.txt` contains:

```text
numpy==2.4.1
Pillow==12.0.0
tqdm==4.67.1
transformers==5.3.0
```

> **Note:** DINOv2 is loaded through `torch.hub`, while CLIP and OWL-ViT are loaded through Hugging Face Transformers. Internet access may therefore be required when these models are used for the first time.

## Data Preparation

The training images should be organized into separate content and style directories.

A recommended directory structure is:

```text
input/
├── Train/
│   ├── Content/
│   └── Style/
│
└── Test/
    ├── Content/
    └── Style/
```

The style transfer network currently uses an input resolution of **224 × 224** during training.

A preprocessing script is provided in `data_preprocess.py`.

Example:

```bash
python data_preprocess.py \
    --source_dir /path/to/source/images \
    --target_dir ./input/Train/Content
```

The preprocessing procedure resizes the image and randomly crops it to 224 × 224.

## Pretrained Models

The pretrained VGG model and trained style transfer checkpoints are not included in this repository.

Place the required model files in:

```text
pre_trained_models/
├── vgg_normalised.pth
└── checkpoint/
    └── checkpoint_40000_epoch.pkl
```

The VGG model is required during training, while the trained checkpoint is required during inference.

## Training

Training is performed using `train.py`.

Example:

```bash
python train.py \
    --content_dir ./input/Train/Content \
    --style_dir ./input/Train/Style \
    --vgg_dir ./pre_trained_models/vgg_normalised.pth \
    --batch_size 4 \
    --epoch 40000 \
    --checkpoint_save_interval 10000 \
    --checkpoint_save_path ./pre_trained_models/checkpoint
```

The default training configuration is:

| Parameter              | Default |
| ---------------------- | ------: |
| Base learning rate     |  `1e-4` |
| Batch size             |     `4` |
| Training iterations    | `40000` |
| Content loss weight    |     `5` |
| Style loss weight      |     `3` |
| Identity loss 1 weight |    `50` |
| Identity loss 2 weight |     `1` |
| Checkpoint interval    | `10000` |
| Loss logging interval  |   `400` |

Checkpoints are saved to:

```text
pre_trained_models/checkpoint/
```

For example:

```text
checkpoint_10000_epoch.pkl
checkpoint_20000_epoch.pkl
checkpoint_30000_epoch.pkl
checkpoint_40000_epoch.pkl
```

## Inference

Inference is performed using `test.py`.

Unlike conventional inference that directly specifies both a content image and a style image, this implementation takes a **content image and a directory of candidate style images**.

The style matching module first selects a semantically appropriate style image from the candidate directory, and the selected content-style pair is then passed to the style transfer network.

Example:

```bash
python test.py \
    --content_path ./input/Test/Content/000000001371.jpg \
    --style_dir ./input/Test/Style \
    --checkpoint_import_path ./pre_trained_models/checkpoint/checkpoint_40000_epoch.pkl \
    --output_dir ./output
```

The inference pipeline is:

```text
Content Image
      │
      ▼
Object Detection
      │
      ▼
Semantic Group Matching
      │
      ▼
CLIP Similarity
      │
      ▼
Style Image Selection
      │
      ▼
Semantic Style Transfer
      │
      ▼
Stylized Image
```

Generated images are saved in:

```text
output/
```

## Main Components

### S2WAT Encoder

S2WAT is used as the visual feature encoder to extract hierarchical representations from the content and style images.

### DINOv2 Semantic Feature Extractor

DINOv2 is used to extract semantic patch features. The semantic features of the content and style images are used to construct semantic similarity matrices that guide feature correspondence during style transfer.

### AdaFormer

AdaFormer establishes correspondences between content and style features. Semantic similarity derived from DINOv2 is incorporated into the attention computation to encourage semantically consistent feature matching.

### Semantic-aware Style Matching

Before style transfer, OWL-ViT is used for object detection and semantic group analysis. CLIP similarity is then used together with semantic information to select an appropriate style image from the candidate style set.

## Output

A successful inference produces a stylized image using the automatically selected style image.

Future versions of this repository will include qualitative examples and comparison results.

## Citation

This repository is associated with a master's thesis on semantic-aware arbitrary image style transfer.

The citation information will be added after the thesis/publication information is finalized.

## Acknowledgements

This project builds upon concepts and components from S2WAT, AdaAttN/AdaFormer, DINOv2, CLIP, and OWL-ViT.

Please refer to the corresponding original works when using or extending this project.
