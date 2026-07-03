# 🌍 TerraFuse
**Multi-Modal SAR–Optical Land Cover Classification with Dual-Branch CNNs**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![Task](https://img.shields.io/badge/Task-Land_Cover_Classification-green)
![Fusion](https://img.shields.io/badge/Architecture-Late_Feature_Fusion-purple)
![License](https://img.shields.io/badge/License-MIT-black)

## Overview
TerraFuse is a PyTorch-based computer vision project that classifies paired Sentinel-1 SAR and Sentinel-2 optical satellite imagery into four terrain categories: **agricultural land**, **barrenland**, **grassland**, and **urban**. 

The system implements a **dual-branch Convolutional Neural Network (CNN)** that performs **late feature-level fusion** to combine learned representations from both modalities.

## Motivation
Single-modality remote sensing has inherent limitations. Sentinel-1 SAR (Synthetic Aperture Radar) can penetrate clouds and capture images at night, but produces noisy, non-photorealistic backscatter images. Sentinel-2 optical imagery is photorealistic and rich in spectral information, but is easily obstructed by weather conditions and clouds. Fusing both modalities enables a more robust classification system that leverages the complementary strengths of radar and optical sensors.

## Results
The metrics below represent the performance of the architectures. 

| Model | Modality | Best Accuracy |
|---|---|---|
| **SAR CNN** | Sentinel-1 (Grayscale) | **98.38%** (Test)* |
| **Optical CNN** | Sentinel-2 (RGB) | **98.08%** (Test)* |
| **Fusion CNN** | SAR + Optical (Late Fusion) | **99.75%** (Validation @ Epoch 4)** |

*\* SAR and Optical test metrics were obtained during initial experiments using legacy dataset splits.*
*\*\* Fusion training was manually interrupted at epoch 4. The 99.75% represents validation accuracy at the time of interruption. A final held-out test accuracy for the fusion model was not evaluated.*

## Architecture
TerraFuse uses a dual-branch CNN with **late feature-level fusion**. Each modality is processed by an independent, modality-specific CNN encoder. The resulting high-dimensional feature vectors are concatenated before being passed to a joint fusion head for final classification.

```text
Sentinel-1 SAR (1-ch, 256×256)         Sentinel-2 Optical (3-ch, 256×256)
         │                                          │
    ┌────▼────┐                              ┌──────▼──────┐
    │ SAR CNN │  5 conv blocks               │ Optical CNN │  5 conv blocks
    │         │  (1→32→64→64→128→128)        │             │  (3→32→64→64→128→128)
    └────┬────┘                              └──────┬──────┘
         │  flatten → 8192-dim                      │  flatten → 8192-dim
         └──────────────────┬───────────────────────┘
                            │  torch.cat → 16384-dim
                     ┌──────▼──────┐
                     │ Fusion Head │  Linear(16384→128) → ReLU → Dropout(0.5) → Linear(128→4)
                     └──────┬──────┘
                            │
                     4-class logits
              (agri · barrenland · grassland · urban)
```
Each encoder convolution block consists of: `Conv2d(3×3, pad=1)` → `BatchNorm2d` → `ReLU` → `MaxPool2d(2×2)`.

### Fusion Strategy
1. **Single-Modal Training**: Both the SAR-only and Optical-only branches are trained to convergence independently.
2. **Warm-Starting**: The feature extractors of the fusion model are initialized with the weights from the pre-trained single-modal models.
3. **Joint Fine-Tuning**: The fusion head (which is initialized randomly) and both branches are trained jointly. Gradients flow through all parameters, allowing the individual branch representations to co-adapt to each other.

## Dataset
The project is built for the Kaggle dataset: [Sentinel-1/2 Image Pairs Segregated by Terrain](https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain).

- **Total Images**: 16,000 paired images (4,000 per class).
- **Classes**: `agri`, `barrenland`, `grassland`, `urban`.
- **Dimensions**: 256×256 pixels.
- **Split**: 70% Train / 15% Validation / 15% Test.
- **Reproducibility**: Splits are deterministic (Seed 42) and based on a canonical unified sample manifest to strictly prevent data leakage across modalities.

## Installation

```bash
# Clone the repository
git clone https://github.com/raghavvats/terrafuse-sar-optical-classification.git
cd terrafuse-sar-optical-classification

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Setup
1. Download the dataset from Kaggle.
2. Set the `TERRAFUSE_DATA_DIR` environment variable to the root of the dataset (the folder containing the class subdirectories).

```bash
export TERRAFUSE_DATA_DIR="/path/to/dataset/v_2"
```

## Training

### Training Configuration
| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Batch Size | 32 |
| Epochs | 20 |
| Loss Function | CrossEntropyLoss |
| Dropout | 0.5 |
| Image Size | 256 × 256 |
| SAR Normalization | `mean=0.5, std=0.5` |
| Optical Normalization | ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) |

### Running Training
Use `main.py` to launch training. Edit the `__main__` block to choose the model:
```python
# In main.py:
sar_main()                       # Train SAR-only model
optical_main()                   # Train Optical-only model
fusion_main(pretrained=True)     # Train fusion model (warm-starts from above)
```
Then run:
```bash
python main.py
```

### Checkpointing
The training loop evaluates the model on the validation set at the end of each epoch and saves the weights to disk (e.g., `best_model_fusion.pth`) only if the validation accuracy improves. If training is interrupted, simply re-run the script; it will automatically load the checkpoint, restore optimizer states, and resume from the last saved epoch.

## Inference
The `inference.py` module provides utilities for making predictions on single image pairs.

```python
from inference import load_model, predict_fusion, print_result

model = load_model("fusion", "best_model_fusion.pth")

result = predict_fusion(
    model,
    sar_path="path/to/sar_image_s1.png",
    opt_path="path/to/optical_image_s2.png"
)

print_result(result)
```

## Limitations
- **Fusion Results**: The reported fusion accuracy is a validation metric from an interrupted run, not a final held-out test metric.
- **Classes**: The system is currently limited to four fundamental terrain classes.
- **Generalization**: As with many remote sensing models trained on specific geographic regions, performance may degrade when applied to fundamentally different geographies or seasonal conditions.

## Future Work
- **Cross-Attention Fusion**: Implementing attention mechanisms between SAR and optical feature maps rather than simple concatenation.
- **Pretrained Encoders**: Replacing the custom CNN backbones with robust architectures like ResNet or EfficientNet.
- **Learning Rate Scheduling**: Adding Cosine Annealing or ReduceLROnPlateau.
- **Modality-Specific Augmentation**: Applying distinct augmentations such as color jitter (optical only) and speckle noise (SAR).
- **Ablation Studies**: Rigorously testing missing-modality robustness (e.g., simulating heavy cloud cover).
- **Interpretability**: Generating Grad-CAM heatmaps to analyze which features drive classification decisions.

## References
- [Sentinel-1/2 Dataset on Kaggle](https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain)
- Dimitrovski et al., *Deep Multimodal Fusion for Semantic Segmentation of Remote Sensing Earth Observation Data*, arXiv:2410.00469

## License
MIT License. See [LICENSE](LICENSE) for details.
