# Vision Transformer (ViT) for CIFAR-100

This project implements a Vision Transformer (ViT) model for image classification on the CIFAR-100 dataset using PyTorch.

## Overview

The Vision Transformer (ViT) is an architecture that applies the transformer model to image classification by splitting images into patches and processing them as sequences. This implementation is specifically designed for the CIFAR-100 dataset, which contains 100 classes of images.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- NumPy
- PIL

## Model Architecture

The ViT model includes:
- Patch embedding layer
- Position embeddings
- Multiple transformer encoder layers
- MLP head for classification

## Dataset

CIFAR-100 consists of:
- 50,000 training images
- 10,000 test images
- 100 classes
- 32x32 pixel RGB images

## Usage

1. Install dependencies:
```bash
pip install torch torchvision numpy pillow
```

2. Run the training:
```bash
python viz_transformer.py
```

## Model Parameters

- Patch size: 4x4
- Hidden dimension: 768
- Number of transformer layers: 12
- Number of attention heads: 12
- MLP dimension: 3072
- Dropout: 0.1

## Training Details

- Optimizer: AdamW
- Learning rate: 3e-4
- Weight decay: 0.01
- Training epochs: 100
- Batch size: 128

## License

MIT License
