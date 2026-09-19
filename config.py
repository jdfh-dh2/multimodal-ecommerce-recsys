"""
Configuration for Multi-Modal E-Commerce Recommendation System
=============================================================
"""

# Model Configuration
MODEL_CONFIG = {
    # DSSM Model
    'dssm': {
        'embedding_dim': 128,
        'hidden_dims': [256, 128, 64],
        'dropout': 0.2,
        'user_feature_dim': 64,
        'item_feature_dim': 128,
        'use_multimodal': True
    },

    # DIN Model
    'din': {
        'embedding_dim': 128,
        'hidden_dims': [200, 80],
        'dropout': 0.2,
        'max_behavior_seq_len': 50,
        'user_feature_dim': 64,
        'num_items': 50000,
        'use_multimodal': True
    },

    # Multi-task Models (ESMM, MMoE)
    'multi_task': {
        'embedding_dim': 128,
        'shared_hidden_dims': [256, 128],
        'task_hidden_dims': [64, 32],
        'dropout': 0.2,
        'input_dim': 448,  # user_feature_dim + embedding_dim * 3
        'num_experts': 4,
        'num_tasks': 2
    }
}

# Training Configuration
TRAIN_CONFIG = {
    'epochs': 10,
    'batch_size': 256,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'gradient_clip': 5.0,
    'use_multimodal': True,
    'auxiliary_weight': 0.2,
    'num_workers': 4
}

# Data Configuration
DATA_CONFIG = {
    'num_users': 10000,
    'num_items': 50000,
    'num_interactions': 100000,
    'max_behavior_seq_len': 50,
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15
}

# Feature Extraction Configuration
FEATURE_CONFIG = {
    'image': {
        'model': 'resnet50',
        'embedding_dim': 2048,
        'image_size': 224
    },
    'text': {
        'model': 'bert-base-uncased',
        'embedding_dim': 768,
        'max_length': 128
    }
}

# A/B Testing Configuration
AB_TEST_CONFIG = {
    'treatment_ratio': 0.5,
    'min_sample_size': 100,
    'significance_level': 0.05,
    'metrics': ['ctr', 'cvr', 'gmv', 'session_length', 'items_purchased']
}

# Evaluation Metrics
EVAL_METRICS = {
    'ctr': {
        'metrics': ['auc', 'accuracy', 'mrr', 'ndcg', 'precision@k'],
        'k_values': [1, 3, 5, 10, 20]
    },
    'cvr': {
        'metrics': ['auc', 'pr_auc', 'mse', 'mae']
    },
    'multitask': {
        'metrics': ['ctr_auc', 'cvr_auc', 'ctcvr_auc']
    }
}
