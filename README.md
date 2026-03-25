# JEPA-Lens

JEPA-Lens is a project focused on the Joint Embedding Predictive Architecture (JEPA), a groundbreaking self-supervised learning approach proposed by Yann LeCun for building more human-like AI systems.

## About JEPA

JEPA aims to create AI models that learn internal models of how the world works, enabling faster learning, better planning, and adaptation to new situations. Unlike traditional methods that predict pixels or tokens directly, JEPA predicts abstract representations, capturing common-sense knowledge.

### I-JEPA: Image Joint Embedding Predictive Architecture

I-JEPA is the first implementation of JEPA for computer vision tasks. It learns by predicting missing parts of images in an abstract representation space, rather than reconstructing pixels. This approach:

- Captures semantic understanding of scenes
- Avoids biases from invariance-based pretraining
- Is computationally efficient (no need for multiple image augmentations)
- Achieves state-of-the-art performance on vision tasks with fewer resources

Key features of I-JEPA:
- Uses a Vision Transformer (ViT) architecture
- Employs multi-block masking strategy
- Predicts high-level object parts with correct pose and position
- Outperforms generative methods on ImageNet classification

For more details, read the [original blog post](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/) by Yann LeCun.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

[To be added - project-specific usage instructions]

## References

- [I-JEPA Paper](https://arxiv.org/abs/2301.08243)
- [I-JEPA Code and Checkpoints](https://github.com/facebookresearch/ijepa)