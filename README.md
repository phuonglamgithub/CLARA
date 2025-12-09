# CLARA: Enhancing Multimodal Sentiment Analysis via Efficient Vision-Language Fusion

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXX-blue)](https://doi.org/10.5281/zenodo.XXXXXX)

**Official Implementation**

> **⚠️ IMPORTANT NOTICE**
> 
> This code is **directly related** to our manuscript currently **under review** at ***The Visual Computer*** (Springer):
> 
> **"CLARA: Enhancing Multimodal Sentiment Analysis via Efficient Vision-Language Fusion"**
> 
> - **Authors**: Phuong Lam, Tuoi Thi Phan, Thien Khai Tran
> - **Status**: Under Review (Resubmission)
> - **Submission ID**: 327916a7-2e80-4286-b6c7-91175346a1e7
> - **Journal**: The Visual Computer (Springer)
> - **Submitted**: December 2025
> 
> **📢 If you use this code or find our work helpful, please cite our paper** (see [Citation](#citation) section below).

---

## 📋 Table of Contents

- [Abstract](#abstract)
- [Key Features](#key-features)
- [Performance Results](#performance-results)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Datasets](#datasets)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## 📝 Abstract

Understanding sentiment in social images requires integrating visual content with short text, where cross-modal conflicts are prevalent. We introduce **CLARA**, a parameter-efficient vision-language framework for multimodal sentiment analysis on image-text pairs. 

CLARA employs lightweight **LoRA adapters** on frozen encoders (CLIP for vision, DeBERTa for text), coupled with **multi-head co-attention** for aligning visual regions and textual spans. A **consistency-verification** step refines the fused representation before classification.

Our approach achieves **state-of-the-art results** on three diverse datasets:
- **MVSA-Single**: 83.04% weighted F1, 83.16% accuracy
- **MVSA-Multiple**: 73.45% weighted F1, 73.51% accuracy
- **HFM** (hate speech detection): 87.82% macro F1

Demonstrating effective generalization while maintaining **parameter efficiency** (only **7.45% trainable parameters**). CLARA significantly improves neutral class prediction and provides well-calibrated predictions under modal disagreement.

### 🔗 Code Availability

**Complete source code** with comprehensive documentation:
- **GitHub Repository**: https://github.com/phuonglamgithub/CLARA
- **Permanent Archive (Zenodo)**: DOI: [`10.5281/zenodo.XXXXXX`](https://doi.org/10.5281/zenodo.XXXXXX)
- **License**: MIT License
- **Documentation**: Installation guides, usage examples, dataset preparation

All implementations are released under the MIT License to facilitate reproduction and extension of our work.

---

## ✨ Key Features

### 🎯 Model Architecture
- **Parameter-Efficient**: Only **7.45% trainable parameters** via LoRA adapters
- **Frozen Encoders**: CLIP-ViT-Base-32 (vision) + DeBERTa-v3-Base (text)
- **Hierarchical Co-Attention**: Multi-head attention for vision-language alignment
- **Dual Verification-Prediction**: Separate pathways for confidence estimation
- **Post-HOC Calibration**: Temperature scaling for improved confidence

### 📊 Datasets Supported
- ✅ **MVSA-Single**: 4,869 image-text pairs, 3 classes (Pos/Neu/Neg)
- ✅ **MVSA-Multiple**: 19,600 samples with multiple annotators
- ✅ **HFM (Hateful Memes)**: 10,000+ memes for hate speech detection

### 🚀 Training Features
- **Class Imbalance Handling**: Focal loss with class-specific gamma
- **Early Stopping**: Patience-based with validation monitoring
- **Mixed Precision**: FP16 training for efficiency
- **Gradient Accumulation**: Support for large batch sizes
- **Comprehensive Logging**: Weights & Biases integration (optional)

---

## 📊 Performance Results

### Comparison with State-of-the-Art

| Dataset | Method | Weighted F1 | Macro F1 | Accuracy | Trainable Params |
|---------|--------|-------------|----------|----------|------------------|
| **MVSA-Single** | Previous SOTA | 79.21% | - | 79.33% | 100% |
| | **CLARA (Ours)** | **83.04%** | **82.85%** | **83.16%** | **7.45%** |
| | *Improvement* | *+3.83%* | - | *+3.83%* | *-92.55%* |
| **MVSA-Multiple** | Previous SOTA | 71.52% | - | 71.78% | 100% |
| | **CLARA (Ours)** | **73.45%** | **61.48%** | **73.51%** | **7.45%** |
| | *Improvement* | *+1.93%* | - | *+1.73%* | *-92.55%* |
| **HFM** | Previous SOTA | - | 86.50% | - | 100% |
| | **CLARA (Ours)** | **87.71%** | **87.82%** | **87.65%** | **7.45%** |
| | *Improvement* | - | *+1.32%* | - | *-92.55%* |

### Key Improvements
- ✅ **+3.83%** accuracy on MVSA-Single over previous SOTA
- ✅ **+6-7%** F1 score improvement on Neutral class
- ✅ **92.55%** reduction in trainable parameters
- ✅ **Well-calibrated** predictions with ECE < 0.05

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM recommended
- GPU with 12GB+ VRAM (Tesla T4, RTX 3090, or better)

### Quick Install

```bash
# Clone repository
git clone https://github.com/phuonglamgithub/CLARA.git
cd CLARA

# Create conda environment
conda create -n clara python=3.8 -y
conda activate clara

# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

**Detailed installation guide**: See [`docs/INSTALLATION.md`](docs/INSTALLATION.md)

---

## 🚀 Quick Start

### 1. Prepare Datasets

Download datasets and organize according to structure in [`data/README.md`](data/README.md):

```bash
# MVSA-Single structure
data/MVSA_Single/
├── data/
│   ├── 1.jpg, 1.txt, ...
│   └── ...
└── labelResultAll.txt

# MVSA-Multiple structure  
data/MVSA_Multiple/
├── data/
│   ├── 10001.jpg, 10001.txt, ...
│   └── ...
└── labelResultAll.txt

# HFM structure
data/HFM/
├── img/
│   └── *.png
├── train.jsonl
├── dev_seen.jsonl
└── test_seen.jsonl
```

### 2. Run Training

```bash
# Train on MVSA-Single
python src/training/train.py --config configs/mvsa_single.yaml

# Train on MVSA-Multiple
python src/training/train.py --config configs/mvsa_multiple.yaml

# Train on HFM
python src/training/train.py --config configs/hfm.yaml
```

### 3. Evaluate Model

```bash
# Evaluate on test set
python src/training/evaluate.py \
    --checkpoint checkpoints/clara_mvsa_single_best.pth \
    --config configs/mvsa_single.yaml
```

### 4. Use Pre-trained Models

```python
from src.models.clara_model import CLARA
from src.utils.inference import predict_sentiment

# Load model
model = CLARA.from_pretrained("checkpoints/clara_mvsa_single_best.pth")

# Predict sentiment
result = predict_sentiment(
    model=model,
    image_path="example.jpg",
    text="This is an amazing view!"
)

print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2%}")
```

---

## 📂 Repository Structure

```
CLARA/
├── 📄 README.md                          # This file
├── 📄 LICENSE                            # MIT License
├── 📄 requirements.txt                   # Python dependencies
├── 📄 CITATION.bib                       # BibTeX citation
├── 📄 .gitignore                         # Git ignore rules
│
├── 📁 src/                               # Source code
│   ├── models/
│   │   ├── clara_model.py               # Main CLARA architecture
│   │   ├── lora_adapters.py             # LoRA implementation
│   │   └── attention.py                 # Co-attention modules
│   ├── data/
│   │   ├── mvsa_dataset.py              # MVSA dataset loaders
│   │   ├── hfm_dataset.py               # HFM dataset loader
│   │   └── preprocessing.py             # Data preprocessing
│   ├── training/
│   │   ├── train.py                     # Training script
│   │   ├── evaluate.py                  # Evaluation script
│   │   └── losses.py                    # Loss functions
│   └── utils/
│       ├── metrics.py                   # Evaluation metrics
│       ├── inference.py                 # Inference utilities
│       └── visualization.py             # Plotting functions
│
├── 📁 configs/                           # Configuration files
│   ├── mvsa_single.yaml                 # MVSA-Single config
│   ├── mvsa_multiple.yaml               # MVSA-Multiple config
│   └── hfm.yaml                         # HFM config
│
├── 📁 notebooks/                         # Jupyter notebooks
│   ├── CLARA_MVSA_Single.ipynb          # MVSA-Single experiments
│   ├── CLARA_MVSA_Multi.ipynb           # MVSA-Multiple experiments
│   └── CLARA_HFM.ipynb                  # HFM experiments
│
├── 📁 data/                              # Dataset directory
│   ├── README.md                        # Dataset guide
│   ├── MVSA_Single/                     # MVSA-Single data
│   ├── MVSA_Multiple/                   # MVSA-Multiple data
│   └── HFM/                             # HFM data
│
├── 📁 docs/                              # Documentation
│   ├── INSTALLATION.md                  # Installation guide
│   ├── USAGE.md                         # Usage guide
│   ├── DATASETS.md                      # Dataset details
│   └── ARCHITECTURE.md                  # Model architecture
│
├── 📁 checkpoints/                       # Pre-trained models
│   └── README.md                        # Model documentation
│
├── 📁 results/                           # Experimental results
│   ├── figures/                         # Visualizations
│   └── metrics/                         # Performance logs
│
└── 📁 scripts/                           # Helper scripts
    ├── download_datasets.sh             # Dataset downloader
    └── run_all_experiments.sh           # Run all experiments
```

---

## 📊 Datasets

### MVSA-Single
- **Size**: 4,869 image-text pairs
- **Classes**: 3 (Positive, Neutral, Negative)
- **Source**: Twitter social media posts
- **Download**: See [`data/README.md`](data/README.md)

### MVSA-Multiple
- **Size**: 19,600 image-text pairs
- **Classes**: 3 (Positive, Neutral, Negative)
- **Annotators**: Multiple annotators per sample
- **Challenge**: Severe class imbalance (15% Neutral)
- **Download**: See [`data/README.md`](data/README.md)

### HFM (Hateful Memes)
- **Size**: 10,000+ memes
- **Classes**: 2 (Hateful, Not Hateful)
- **Challenge**: Requires multimodal reasoning
- **Source**: Facebook AI Research
- **Download**: https://ai.facebook.com/tools/hatefulmemes/

**Detailed dataset information**: See [`data/README.md`](data/README.md)

---

## 📖 Citation

### 📢 Please Cite Our Work

If you use CLARA in your research or find this repository helpful, **please cite our paper**:

#### Paper Citation (Under Review)

```bibtex
@article{lam2025clara,
  title={CLARA: Enhancing Multimodal Sentiment Analysis via Efficient Vision-Language Fusion},
  author={Lam, Phuong and Phan, Tuoi Thi and Tran, Thien Khai},
  journal={The Visual Computer},
  year={2025},
  publisher={Springer},
  note={Under review. Submission ID: 327916a7-2e80-4286-b6c7-91175346a1e7},
  url={https://github.com/phuonglamgithub/CLARA}
}
```

#### Code Repository Citation

```bibtex
@software{lam2025clara_code,
  author={Lam, Phuong and Phan, Tuoi Thi and Tran, Thien Khai},
  title={CLARA: Official Implementation - Parameter-Efficient Vision-Language Fusion for Multimodal Sentiment Analysis},
  year={2025},
  publisher={Zenodo},
  doi={10.5281/zenodo.XXXXXX},
  url={https://github.com/phuonglamgithub/CLARA},
  note={Source code for paper under review at The Visual Computer}
}
```

### Related Citations

If you use specific components, please also consider citing:

```bibtex
@article{hu2021lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and others},
  journal={arXiv preprint arXiv:2106.09685},
  year={2021}
}

@inproceedings{radford2021learning,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and others},
  booktitle={ICML},
  year={2021}
}
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Phuong Lam, Tuoi Thi Phan, Thien Khai Tran

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## 📧 Contact

### Authors

**Phuong Lam**, **Tuoi Thi Phan**, **Thien Khai Tran**

- **Affiliation**: Ho Chi Minh City University of Technology (HCMUT)
- **Email**: thientk@huit.edu.vn
- **GitHub**: https://github.com/phuonglamgithub/CLARA

### Support

For questions, issues, or collaboration:
- 📧 **Email**: thientk@huit.edu.vn
- 🐛 **Issues**: [GitHub Issues](https://github.com/phuonglamgithub/CLARA/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/phuonglamgithub/CLARA/discussions)

---

## 🙏 Acknowledgments

- This work is submitted to **The Visual Computer** (Springer) journal
- Pre-trained models: OpenAI CLIP, Microsoft DeBERTa
- Datasets: MVSA, Facebook AI Research (HFM)
- Framework: PyTorch, HuggingFace Transformers, PEFT

### Related Works

Our work builds upon recent advances in multimodal sentiment analysis. See also:

- **CCMA**: CapsNet for audio-video sentiment analysis using cross-modal attention. *The Visual Computer*, 2025, 41(3): 1609-1620.
- **CTHFNet**: Contrastive translation and hierarchical fusion network for text-video-audio sentiment analysis. *The Visual Computer*, 2025, 41(7): 4405-4418.

---

## 📊 Repository Statistics

![GitHub Stars](https://img.shields.io/github/stars/phuonglamgithub/CLARA?style=social)
![GitHub Forks](https://img.shields.io/github/forks/phuonglamgithub/CLARA?style=social)
![GitHub Issues](https://img.shields.io/github/issues/phuonglamgithub/CLARA)
![GitHub Pull Requests](https://img.shields.io/github/issues-pr/phuonglamgithub/CLARA)

---

<div align="center">

**⭐ If you find this work useful, please star the repository! ⭐**

**📄 Paper under review at The Visual Computer (Springer)**

**Made with ❤️ by the CLARA Team**

</div>
