# MultiModal-RecSys: 多模态电商推荐系统

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

一个综合性的多模态电商推荐系统，结合视觉和文本产品理解与顺序用户行为建模，实现个性化推荐。

## 🌟 功能特点

- **多模态内容理解**：结合ResNet50图像特征和BERT文本嵌入
- **顺序用户建模**：DIN风格注意力机制处理行为序列
- **多任务学习**：ESMM和MMoE架构联合优化CTR/CVR
- **生产级A/B测试**：内置统计显著性测试框架
- **多模型支持**：DSSM、DIN、ESMM、MMoE实现

## 📋 目录

- [功能特点](#-功能特点)
- [快速开始](#-快速开始)
- [项目结构](#-项目结构)
- [模型介绍](#-模型介绍)
- [评估系统](#-评估系统)
- [文档说明](#-文档说明)

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/fish-fly-coder/MultiModal-RecSys.git
cd MultiModal-RecSys
pip install torch torchvision transformers scikit-learn scipy pandas numpy
```

### 训练

```bash
# 训练所有模型
python train.py --model all --epochs 10

# 训练特定模型
python train.py --model dssm --epochs 5

# 自定义配置
python train.py --model din --batch_size 128 --lr 0.0005
```

### 评估

```python
from evaluation.ab_testing import RecommendationEvaluator
from data.data_loader import create_synthetic_data, split_data, create_dataloaders

# 创建数据
df = create_synthetic_data(num_users=5000, num_items=10000)
train_df, val_df, test_df = split_data(df)
loaders = create_dataloaders(train_df, val_df, test_df, config)

# 评估模型
evaluator = RecommendationEvaluator()
metrics = evaluator.compute_ctr_metrics(predictions, labels)
print(f"AUC: {metrics['auc']:.4f}, MRR: {metrics['mrr']:.4f}")
```

## 📁 项目结构

```
multimodal-ecommerce-recsys/
├── models/                    # 模型实现
│   ├── __init__.py
│   ├── dssm.py               # 双塔DSSM模型
│   ├── din.py                # 深度兴趣网络
│   └── multi_task.py         # ESMM和MMoE模型
├── features/                  # 特征提取
│   ├── __init__.py
│   └── multimodal_encoder.py # 图像和文本编码器
├── data/                      # 数据加载
│   ├── __init__.py
│   └── data_loader.py        # 数据集和数据加载器
├── evaluation/                # 评估框架
│   ├── __init__.py
│   └── ab_testing.py         # A/B测试和指标
├── train.py                   # 主训练脚本
├── config.py                  # 配置文件
├── README.md                  # 英文文档
├── README_CH.md              # 中文文档
└── detailed-explanation.md   # 技术细节
```

## 🧠 模型介绍

### DSSM (深度结构化语义模型)
双塔架构，在共享潜在空间学习用户和物品表示。

```python
from models.dssm import DSSMModel
model = DSSMModel(config)
similarity, user_emb, item_emb = model(user_features, item_features)
```

### DIN (深度兴趣网络)
基于注意力的模型，动态学习用户行为历史与目标物品的相关性。

```python
from models.din import DINModel
model = DINModel(config)
ctr_score, attention_weights = model(user_features, item_ids, behavior_seq, behavior_length)
```

### ESMM (全空间多任务模型)
联合CTR/CVR预测，避免数据稀疏问题。

### MMoE (多门混合专家)
带任务特定专家门的多任务学习。

## 📊 评估系统

### A/B测试

```python
from evaluation.ab_testing import ABTestEvaluator, simulate_ab_test_data

# 模拟A/B测试
assignments, tracker = simulate_ab_test_data(num_users=1000)

# 评估
evaluator = ABTestEvaluator("test", "control_model", "treatment_model")
evaluator.tracker = tracker
report = evaluator.evaluate(metrics=['ctr', 'cvr', 'gmv'])
```

### 评估指标

| 指标 | 描述 | 目标 |
|--------|-------------|--------|
| AUC | ROC曲线下面积 | > 0.85 |
| CTR | 点击率 | +15%提升 |
| CVR | 转化率 | +10%提升 |
| MRR | 平均倒数排名 | > 0.6 |

## 📖 文档

- [英文README](README.md) - 快速开始和概览
- [中文README](README_CH.md) - 中文说明文档
- [详细技术文档](detailed-explanation.md) - 技术细节说明

## 🎯 职位匹配

本项目展示与国际电商算法岗位相关的技能：

| 职位要求 | 项目覆盖 |
|----------------|------------------|
| 多模态内容理解 | ResNet50 + BERT融合 |
| 用户行为建模 | DIN注意力机制 |
| CTR/CVR优化 | ESMM、MMoE多任务学习 |
| 推荐系统 | DSSM双塔架构 |
| A/B测试框架 | 内置统计测试 |

## 📝 许可证

MIT许可证 - 详见LICENSE文件。

## 🙏 致谢

本项目参考了以下架构：
- NVIDIA Merlin Transformers4Rec
- 腾讯ESMM
- 阿里巴巴DIN
- 谷歌MMoE

---

**作者**: fish-fly-coder
**邮箱**: WorthingtonManchini533@outlook.com