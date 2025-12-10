# CLARA: Enhancing Multimodal Sentiment Analysis via Efficient Vision-Language Fusion

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17862924.svg)](https://doi.org/10.5281/zenodo.17862924)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## ⚠️ IMPORTANT NOTICE

**This code is directly related to our manuscript currently under review at The Visual Computer (Springer).**

**If you use this code in your research, please cite our paper:**

```bibtex
@article{lam2025clara,
  title={CLARA: Enhancing Multimodal Sentiment Analysis via Efficient Vision-Language Fusion},
  author={Lam, Phuong and Phan, Tuoi Thi and Tran, Thien Khai},
  journal={The Visual Computer},
  year={2025},
  publisher={Springer},
  note={Under review. Submission ID: 327916a7-2e80-4286-b6c7-91175346a1e7},
  doi={10.5281/zenodo.17862924}
}
```

**Paper Status:**
- **Journal:** The Visual Computer (Springer)
- **Status:** Under Review (Resubmission)
- **Submission ID:** 327916a7-2e80-4286-b6c7-91175346a1e7
- **Submitted:** December 2025

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Performance](#performance)
- [Installation](#installation)
- [Datasets](#datasets)
- [Pre-trained Models](#pre-trained-models)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Citation](#citation)
- [Contact](#contact)

---

## 🎯 Overview

**CLARA** (Co-attention Learning for Robust multimodal sentiment Analysis) is a parameter-efficient vision-language framework for multimodal sentiment analysis on social media image-text pairs.

### Why CLARA?

CLARA achieves **state-of-the-art performance** while updating only **7.45% of parameters**:

- ✅ **83.16% accuracy** on MVSA-Single (+3.83% over previous SOTA)
- ✅ **73.51% accuracy** on MVSA-Multiple (+0.22% over previous SOTA)
- ✅ **88.13% accuracy** on HFM hate speech detection (+1.03% over previous SOTA)
- ✅ **51.9% faster training** compared to full fine-tuning
- ✅ **62.5% lower memory usage** (6GB vs 16GB peak GPU memory)

---

## 🏗️ Architecture

CLARA integrates six key components for efficient and effective multimodal sentiment analysis:

![CLARA Architecture](Architect_CLARA.jpg)

### Components:

1. **Vision Encoder**: CLIP ViT-B/16 with LoRA adapters (r=8, α=16)
2. **Text Encoder**: DeBERTa-base with hybrid LoRA + selective unfreezing
3. **Projection Layers**: Map both modalities to shared 512-dimensional space
4. **Co-Attention Fusion**: Two-layer bidirectional multi-head attention (8 heads)
5. **Verification Module**: Computes modality consensus from unimodal predictions
6. **Feedback Module**: Refines final prediction using disagreement signal

### Key Innovation:

Unlike traditional fusion approaches, CLARA's **bidirectional co-attention** creates explicit associations between visual regions and textual spans through symmetric vision↔text and text↔vision processing paths.

---

## ✨ Key Features

### 1. Parameter Efficiency
- **Only 7.45% trainable parameters** (21.27M / 285.47M total)
- **LoRA rank r=8** applied to attention projections
- **92.2% parameter reduction** vs full fine-tuning

### 2. Cross-Modal Reasoning
- **Bidirectional co-attention** prevents text bias
- **8 attention heads** for diverse alignment patterns
- **Explicit image-phrase associations**

### 3. Consensus Modeling
- **Verification module** computes agreement signal
- **Feedback mechanism** integrates consensus
- **Enhanced neutral class** prediction (+8.34% F1)

### 4. Robust Performance
- **State-of-the-art** on 3 diverse benchmarks
- **Strong generalization** across sentiment classes
- **Well-calibrated predictions**

---

## 📊 Performance

### Main Results

| Dataset | Metric | CLARA | Previous SOTA | Improvement |
|---------|--------|-------|---------------|-------------|
| **MVSA-Single** | Accuracy | **83.16%** | 79.33% | +3.83% |
| | Weighted F1 | **83.04%** | 77.51% | +5.53% |
| **MVSA-Multiple** | Accuracy | **73.51%** | 73.29% | +0.22% |
| | Weighted F1 | **73.45%** | 70.06% | +3.39% |
| **HFM** | Accuracy | **88.13%** | 87.10% | +1.03% |
| | Macro F1 | **87.82%** | 86.62% | +1.20% |

### Efficiency Comparison

| Method | Trainable | Train Time | Memory | F1 Score |
|--------|-----------|------------|--------|----------|
| Full Fine-tuning | 272.0M (100%) | 1.35h | 16GB | 76.50% |
| **CLARA (LoRA)** | **21.27M (7.45%)** | **0.65h** | **6GB** | **83.04%** |
| **Improvement** | **-92.2%** | **-51.9%** | **-62.5%** | **+6.54%** |

---

## 🚀 Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU support)
- 8GB+ GPU memory

### Quick Install

```bash
# Clone repository
git clone https://github.com/phuonglamgithub/CLARA.git
cd CLARA

# Install dependencies
pip install -r requirements.txt
```

### Detailed Installation

#### Method 1: Conda (Recommended)

```bash
# Create conda environment
conda create -n clara python=3.8
conda activate clara

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install transformers==4.30.0 peft==0.4.0
pip install pillow numpy pandas matplotlib seaborn scikit-learn tqdm
```

#### Method 2: pip + virtualenv

```bash
# Create virtual environment
python -m venv clara_env
source clara_env/bin/activate  # Windows: clara_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📊 Datasets

### Download Datasets

All datasets are available on Google Drive:

🔗 **[Download Link: CLARA Datasets](https://drive.google.com/drive/folders/15Dcqsm1OvJbU9_Ok1ylbK_widCekjK0O)**

**Contents:**
- `data/MVSA-Single/` - MVSA-Single dataset
- `data/MVSA-Multiple/` - MVSA-Multiple dataset  
- `data/HFM/` - Hateful Memes dataset

### Dataset Structure

After downloading, organize datasets as follows:

```
CLARA/
├── data/
│   ├── MVSA-Single/
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── annotations.csv
│   │   ├── val/
│   │   └── test/
│   │
│   ├── MVSA-Multiple/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   │
│   └── HFM/
│       ├── train/
│       ├── val/
│       └── test/
```

### Dataset Statistics

#### MVSA-Single
- **Total:** 2,592 samples (after cleaning)
- **Classes:** Positive (1,398), Neutral (470), Negative (724)
- **Train/Val/Test:** 1,813 / 387 / 392

#### MVSA-Multiple
- **Total:** 3,555 samples (after filtering)
- **Classes:** Positive (2,173), Neutral (1,268), Negative (114)
- **Train/Val/Test:** 2,487 / 532 / 536
- **Challenge:** Extremely imbalanced (3.2% negative)

#### HFM (Hateful Memes)
- **Total:** 24,635 samples
- **Classes:** Hateful (13,988), Non-hateful (10,647)
- **Train/Val/Test:** 19,816 / 2,410 / 2,409

---

## 🎯 Pre-trained Models

### Download Model Checkpoints

Pre-trained CLARA models are available on Google Drive:

🔗 **[Download Link: Model Checkpoints](https://drive.google.com/drive/folders/15Dcqsm1OvJbU9_Ok1ylbK_widCekjK0O)**

**Available Models:**
- `checkpoints/clara_mvsa_single.pt` - Trained on MVSA-Single
- `checkpoints/clara_mvsa_multi.pt` - Trained on MVSA-Multiple
- `checkpoints/clara_hfm.pt` - Trained on HFM

### Checkpoint Structure

```
CLARA/
├── checkpoints/
│   ├── clara_mvsa_single.pt
│   ├── clara_mvsa_multi.pt
│   └── clara_hfm.pt
```

### Model Specifications

| Model | Dataset | Size | Accuracy | F1 Score |
|-------|---------|------|----------|----------|
| `clara_mvsa_single.pt` | MVSA-Single | ~85 MB | 83.16% | 83.04% |
| `clara_mvsa_multi.pt` | MVSA-Multiple | ~85 MB | 73.51% | 73.45% |
| `clara_hfm.pt` | HFM | ~85 MB | 88.13% | 87.82% |

---

## ⚡ Quick Start

### 1. Load Pre-trained Model

```python
import torch
from clara import CLARAModel
from PIL import Image

# Load pre-trained model
model = CLARAModel.from_pretrained("checkpoints/clara_mvsa_single.pt")
model.eval()

# Prepare input
image = Image.open("path/to/image.jpg")
text = "Your caption here"

# Predict sentiment
with torch.no_grad():
    prediction = model.predict(image, text)

print(f"Sentiment: {prediction['label']}")
print(f"Confidence: {prediction['confidence']:.2%}")
print(f"Probabilities: {prediction['probs']}")
```

### 2. Batch Prediction

```python
from clara import CLARAModel
from PIL import Image

# Load model
model = CLARAModel.from_pretrained("checkpoints/clara_mvsa_single.pt")

# Batch data
images = [Image.open(f"image_{i}.jpg") for i in range(10)]
texts = ["Caption 1", "Caption 2", ..., "Caption 10"]

# Batch predict
predictions = model.predict_batch(images, texts, batch_size=8)

for i, pred in enumerate(predictions):
    print(f"Sample {i}: {pred['label']} ({pred['confidence']:.2%})")
```

### 3. Evaluate on Test Set

```bash
# Evaluate MVSA-Single
python evaluate.py \
  --model_path checkpoints/clara_mvsa_single.pt \
  --data_dir data/MVSA-Single \
  --split test \
  --output_file results/mvsa_single_results.json

# Evaluate MVSA-Multiple
python evaluate.py \
  --model_path checkpoints/clara_mvsa_multi.pt \
  --data_dir data/MVSA-Multiple \
  --split test \
  --output_file results/mvsa_multi_results.json

# Evaluate HFM
python evaluate.py \
  --model_path checkpoints/clara_hfm.pt \
  --data_dir data/HFM \
  --split test \
  --output_file results/hfm_results.json
```

---

## 🎓 Training

### Train from Scratch

```bash
# Train on MVSA-Single
python train.py \
  --config configs/clara_mvsa_single.yaml \
  --data_dir data/MVSA-Single \
  --output_dir outputs/clara-mvsa-single \
  --gpu 0

# Train on MVSA-Multiple with oversampling
python train.py \
  --config configs/clara_mvsa_multiple.yaml \
  --data_dir data/MVSA-Multiple \
  --output_dir outputs/clara-mvsa-multiple \
  --oversample_negative 9.44 \
  --gpu 0

# Train on HFM
python train.py \
  --config configs/clara_hfm.yaml \
  --data_dir data/HFM \
  --output_dir outputs/clara-hfm \
  --gpu 0
```

### Training Configuration

Example: `configs/clara_mvsa_single.yaml`

```yaml
model:
  vision_encoder: "openai/clip-vit-base-patch16"
  text_encoder: "microsoft/deberta-v3-base"
  lora_rank: 8
  lora_alpha: 16
  lora_dropout: 0.05
  num_attention_heads: 8
  num_attention_layers: 2
  hidden_dim: 512

training:
  num_epochs: 50
  batch_size: 32
  learning_rate: 2e-5
  weight_decay: 0.01
  warmup_ratio: 0.1
  early_stopping_patience: 10
  
optimization:
  optimizer: "adamw"
  scheduler: "cosine"
  max_grad_norm: 1.0
```

### Multi-GPU Training

```bash
# Distributed training on 4 GPUs
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  train.py \
  --config configs/clara_mvsa_single.yaml \
  --data_dir data/MVSA-Single \
  --output_dir outputs/clara-mvsa-single-multigpu
```

---

## 📈 Evaluation

### Standard Evaluation

```python
from clara import CLARAModel, evaluate_model

# Load model
model = CLARAModel.from_pretrained("checkpoints/clara_mvsa_single.pt")

# Evaluate
results = evaluate_model(
    model=model,
    data_dir="data/MVSA-Single",
    split="test",
    batch_size=32
)

# Print results
print(f"Accuracy: {results['accuracy']:.2%}")
print(f"Weighted F1: {results['weighted_f1']:.2%}")
print(f"Macro F1: {results['macro_f1']:.2%}")
print("\nPer-class metrics:")
for label, metrics in results['per_class'].items():
    print(f"{label}: P={metrics['precision']:.2%}, R={metrics['recall']:.2%}, F1={metrics['f1']:.2%}")
```

### Generate Confusion Matrix

```python
from clara.visualization import plot_confusion_matrix

plot_confusion_matrix(
    model=model,
    data_dir="data/MVSA-Single",
    split="test",
    save_path="results/confusion_matrix_mvsa_single.png"
)
```

### Attention Visualization

```python
from clara.visualization import visualize_attention

# Visualize attention for specific example
visualize_attention(
    model=model,
    image_path="data/MVSA-Single/test/images/example.jpg",
    text="This is a test caption",
    save_path="results/attention_visualization.png"
)
```

---

## 📸 Results

### Confusion Matrices

All confusion matrices and performance visualizations are available in the `results/` folder:

🔗 **[Download Link: Results Images](https://drive.google.com/drive/folders/15Dcqsm1OvJbU9_Ok1ylbK_widCekjK0O)**

**Available Visualizations:**
- `results/confusion_matrix_mvsa_single.png`
- `results/confusion_matrix_mvsa_multiple.png`
- `results/confusion_matrix_hfm.png`
- `results/attention_visualization_samples.png`
- `results/performance_comparison.png`

### Results Structure

```
CLARA/
├── results/
│   ├── confusion_matrix_mvsa_single.png
│   ├── confusion_matrix_mvsa_multiple.png
│   ├── confusion_matrix_hfm.png
│   ├── attention_visualization_samples.png
│   ├── performance_comparison.png
│   └── per_class_performance.png
```

---

## 📖 Citation

If you use CLARA in your research, please cite our paper:

### Paper Citation

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

### Code Citation

```bibtex
@software{lam2025clara_code,
  author={Lam, Phuong and Phan, Tuoi Thi and Tran, Thien Khai},
  title={CLARA: Official Implementation},
  year={2025},
  publisher={Zenodo},
  doi={10.5281/zenodo.17862924},
  url={https://github.com/phuonglamgithub/CLARA}
}
```

---

## 📞 Contact

### Authors

- **Phuong Lam**
  - Email: lamphuong.ict89@gmail.com
  - Affiliation: HUFLIT, Vietnam

- **Tuoi Thi Phan**
  - Email: pttuoi@ntt.edu.vn
  - Affiliation: Nguyen Tat Thanh University, Vietnam

- **Thien Khai Tran** (Corresponding Author)
  - Email: thientk@huit.edu.vn
  - Affiliation: HUIT, Vietnam

### Support

- **GitHub Issues:** https://github.com/phuonglamgithub/CLARA/issues
- **Email:** thientk@huit.edu.vn

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Links

- **GitHub:** https://github.com/phuonglamgithub/CLARA
- **Paper:** The Visual Computer (Under Review)
- **DOI:** https://doi.org/10.5281/zenodo.17862924
- **Datasets & Models:** https://drive.google.com/drive/folders/15Dcqsm1OvJbU9_Ok1ylbK_widCekjK0O

---

<div align="center">

**Made with ❤️ by the CLARA Team**

**Parameter-Efficient • State-of-the-Art • Open Source**

</div>
