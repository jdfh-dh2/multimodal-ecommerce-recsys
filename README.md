# MultiModal-RecSys: Multi-Modal E-Commerce Recommendation System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A comprehensive multi-modal e-commerce recommendation system that combines visual and textual product understanding with sequential user behavior modeling for personalized recommendations.

## 🌟 Features

- **Multi-Modal Content Understanding**: Combines ResNet50 image features and BERT text embeddings
- **Sequential User Modeling**: DIN-style attention mechanism for behavior sequence
- **Multi-Task Learning**: Joint CTR/CVR optimization with ESMM and MMoE architectures
- **Production-Ready A/B Testing**: Built-in statistical significance testing framework
- **Multiple Model Support**: DSSM, DIN, ESMM, MMoE implementations

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Models](#-models)
- [Evaluation](#-evaluation)
- [Documentation](#-documentation)

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/fish-fly-coder/MultiModal-RecSys.git
cd MultiModal-RecSys
pip install torch torchvision transformers scikit-learn scipy pandas numpy
```

### Training

```bash
# Train all models
python train.py --model all --epochs 10

# Train specific model
python train.py --model dssm --epochs 5

# Custom configuration
python train.py --model din --batch_size 128 --lr 0.0005
```

### Evaluation

```python
from evaluation.ab_testing import RecommendationEvaluator
from data.data_loader import create_synthetic_data, split_data, create_dataloaders

# Create data
df = create_synthetic_data(num_users=5000, num_items=10000)
train_df, val_df, test_df = split_data(df)
loaders = create_dataloaders(train_df, val_df, test_df, config)

# Evaluate model
evaluator = RecommendationEvaluator()
metrics = evaluator.compute_ctr_metrics(predictions, labels)
print(f"AUC: {metrics['auc']:.4f}, MRR: {metrics['mrr']:.4f}")
```

## 📁 Project Structure

```
multimodal-ecommerce-recsys/
├── models/                    # Model implementations
│   ├── __init__.py
│   ├── dssm.py               # Dual-tower DSSM model
│   ├── din.py                # Deep Interest Network
│   └── multi_task.py         # ESMM & MMoE models
├── features/                  # Feature extraction
│   ├── __init__.py
│   └── multimodal_encoder.py # Image & text encoders
├── data/                      # Data loading
│   ├── __init__.py
│   └── data_loader.py        # Dataset & dataloaders
├── evaluation/                # Evaluation framework
│   ├── __init__.py
│   └── ab_testing.py         # A/B testing & metrics
├── train.py                   # Main training script
├── config.py                  # Configuration
├── README.md                  # English documentation
├── README_CH.md              # Chinese documentation
└── detailed-explanation.md   # Technical details
```

## 🧠 Models

### DSSM (Deep Structured Semantic Model)
Dual-tower architecture for learning user and item representations in a shared latent space.

```python
from models.dssm import DSSMModel
model = DSSMModel(config)
similarity, user_emb, item_emb = model(user_features, item_features)
```

### DIN (Deep Interest Network)
Attention-based model that dynamically learns the relevance of user behavior history to target items.

```python
from models.din import DINModel
model = DINModel(config)
ctr_score, attention_weights = model(user_features, item_ids, behavior_seq, behavior_length)
```

### ESMM (Entire Space Multi-Task Model)
Joint CTR/CVR prediction avoiding data sparsity issues.

### MMoE (Multi-Gate Mixture-of-Experts)
Multi-task learning with task-specific expert gates.

## 📊 Evaluation

### A/B Testing

```python
from evaluation.ab_testing import ABTestEvaluator, simulate_ab_test_data

# Simulate A/B test
assignments, tracker = simulate_ab_test_data(num_users=1000)

# Evaluate
evaluator = ABTestEvaluator("test", "control_model", "treatment_model")
evaluator.tracker = tracker
report = evaluator.evaluate(metrics=['ctr', 'cvr', 'gmv'])
```

### Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| AUC | Area Under ROC Curve | > 0.85 |
| CTR | Click-Through Rate | +15% lift |
| CVR | Conversion Rate | +10% lift |
| MRR | Mean Reciprocal Rank | > 0.6 |

## 📖 Documentation

- [English README](README.md) - Quick start and overview
- [Chinese README](README_CH.md) - 中文说明文档
- [Detailed Explanation](detailed-explanation.md) - Technical documentation

## 🎯 JD Alignment

This project demonstrates skills relevant to international e-commerce algorithm positions:

| JD Requirement | Project Coverage |
|----------------|------------------|
| Multi-modal content understanding | ResNet50 + BERT fusion |
| User behavior modeling | DIN attention mechanism |
| CTR/CVR optimization | ESMM, MMoE multi-task learning |
| Recommendation systems | DSSM dual-tower architecture |
| A/B testing framework | Built-in statistical testing |

## 📝 License

MIT License - See LICENSE file for details.
