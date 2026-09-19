"""
Multi-Task Learning Model for CTR/CVR Prediction
=================================================

This module implements multi-task learning for joint optimization
of Click-Through Rate (CTR) and Conversion Rate (CVR) prediction.

Reference: Tencent "Entire Space Multi-Task Modeling" (ESMM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedBottom(nn.Module):
    """
    Shared Bottom multi-task learning architecture.
    Common representations are learned in the bottom layers,
    then split into task-specific towers.
    """

    def __init__(self, config):
        super(SharedBottom, self).__init__()

        self.config = config
        self.embedding_dim = config.get('embedding_dim', 128)
        self.shared_hidden_dims = config.get('shared_hidden_dims', [256, 128])
        self.task_hidden_dims = config.get('task_hidden_dims', [64, 32])
        self.dropout = config.get('dropout', 0.2)

        # Input projection
        self.input_projection = nn.Linear(
            config.get('input_dim', 256),
            self.shared_hidden_dims[0]
        )

        # Shared bottom layers
        shared_layers = []
        prev_dim = self.shared_hidden_dims[0]
        for hidden_dim in self.shared_hidden_dims[1:]:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            prev_dim = hidden_dim

        self.shared_bottom = nn.Sequential(*shared_layers)

        # Task-specific towers
        self.ctr_tower = self._build_task_tower(prev_dim, self.task_hidden_dims, 1)
        self.cvr_tower = self._build_task_tower(prev_dim, self.task_hidden_dims, 1)

    def _build_task_tower(self, input_dim, hidden_dims, output_dim):
        """Build a task-specific tower."""
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self, features):
        """
        Forward pass.

        Args:
            features: Input features (batch_size, input_dim)

        Returns:
            ctr_pred: CTR predictions (batch_size, 1)
            cvr_pred: CVR predictions (batch_size, 1)
        """
        # Shared representation
        shared_repr = self.input_projection(features)
        shared_repr = self.shared_bottom(shared_repr)

        # Task-specific predictions
        ctr_pred = torch.sigmoid(self.ctr_tower(shared_repr))
        cvr_pred = torch.sigmoid(self.cvr_tower(shared_repr))

        return ctr_pred, cvr_pred


class ESMM(nn.Module):
    """
    Entire Space Multi-Task Model (ESMM).

    Jointly models CTR and CVR in the entire space,
    avoiding the data sparsity problem of CVR.

    Reference: "Entire Space Multi-Task Modeling" by Tencent
    """

    def __init__(self, config):
        super(ESMM, self).__init__()

        self.config = config
        self.embedding_dim = config.get('embedding_dim', 128)
        self.dropout = config.get('dropout', 0.2)

        # Embedding layer for categorical features
        self.embedding = nn.Embedding(
            num_embeddings=config.get('vocab_size', 100000),
            embedding_dim=self.embedding_dim
        )

        # User behavior sequence encoder
        self.sequence_encoder = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.embedding_dim,
            num_layers=2,
            batch_first=True,
            dropout=self.dropout
        )

        # Attention pooling for sequence
        self.attention_pooling = nn.Sequential(
            nn.Linear(self.embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Feature fusion
        fusion_dim = config.get('user_feature_dim', 64) + self.embedding_dim * 3
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        # CTR tower
        self.ctr_tower = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 1)
        )

        # CVR tower
        self.cvr_tower = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 1)
        )

    def forward(self, user_features, item_ids, behavior_seq, behavior_mask):
        """
        Forward pass.

        Args:
            user_features: User profile features
            item_ids: Item IDs
            behavior_seq: User behavior sequence
            behavior_mask: Mask for valid behavior items

        Returns:
            ctr_pred: CTR predictions
            cvr_pred: CVR predictions
            ctcvr_pred: CTCVR predictions (CTR * CVR)
        """
        # Embed item IDs
        item_emb = self.embedding(item_ids)  # (batch, embedding_dim)

        # Encode behavior sequence
        seq_encoded, _ = self.sequence_encoder(behavior_seq)  # (batch, seq_len, hidden)

        # Attention pooling
        attention_scores = self.attention_pooling(seq_encoded)  # (batch, seq_len, 1)
        attention_scores = attention_scores.masked_fill(behavior_mask.unsqueeze(-1) == 0, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=1)
        seq_pooled = torch.sum(attention_weights * seq_encoded, dim=1)  # (batch, hidden)

        # Concatenate all features
        fused_features = torch.cat([
            user_features,
            item_emb,
            seq_pooled,
            item_emb * seq_pooled  # Interaction
        ], dim=-1)

        # Fusion
        shared_repr = self.fusion_layer(fused_features)

        # Task-specific predictions
        ctr_pred = self.ctr_tower(shared_repr)
        cvr_pred = self.cvr_tower(shared_repr)

        # CTCVR = CTR * CVR (for training signal)
        ctcvr_pred = ctr_pred + cvr_pred - ctr_pred * cvr_pred  # Approximation

        return ctr_pred, cvr_pred, ctcvr_pred


class MMoE(nn.Module):
    """
    Multi-Gate Mixture-of-Experts (MMoE) for multi-task learning.

    Uses multiple experts with task-specific gates to dynamically
    adjust the contribution of each expert to each task.

    Reference: "Modeling Task Relationships in Multi-Task Learning" by Google
    """

    def __init__(self, config):
        super(MMoE, self).__init__()

        self.config = config
        self.embedding_dim = config.get('embedding_dim', 128)
        self.num_experts = config.get('num_experts', 4)
        self.num_tasks = config.get('num_tasks', 2)  # CTR and CVR
        self.expert_hidden_dims = config.get('expert_hidden_dims', [128, 64])
        self.tower_hidden_dims = config.get('tower_hidden_dims', [32])
        self.dropout = config.get('dropout', 0.2)

        # Input dimension
        input_dim = config.get('input_dim', 256)

        # Experts (shared knowledge)
        self.experts = nn.ModuleList([
            self._build_expert(input_dim, self.expert_hidden_dims, self.embedding_dim)
            for _ in range(self.num_experts)
        ])

        # Task-specific gates
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, self.num_experts),
                nn.Softmax(dim=-1)
            )
            for _ in range(self.num_tasks)
        ])

        # Task towers
        self.towers = nn.ModuleList([
            self._build_tower(self.embedding_dim, self.tower_hidden_dims, 1)
            for _ in range(self.num_tasks)
        ])

    def _build_expert(self, input_dim, hidden_dims, output_dim):
        """Build an expert network."""
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def _build_tower(self, input_dim, hidden_dims, output_dim):
        """Build a task tower."""
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self, features):
        """
        Forward pass.

        Args:
            features: Input features (batch_size, input_dim)

        Returns:
            predictions: List of task predictions
        """
        # Compute expert outputs
        expert_outputs = [expert(features) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch, num_experts, embedding_dim)

        predictions = []

        for i in range(self.num_tasks):
            # Compute gate weights
            gate_weights = self.gates[i](features)  # (batch, num_experts)

            # Weighted sum of expert outputs
            gate_weights = gate_weights.unsqueeze(-1)  # (batch, num_experts, 1)
            task_repr = torch.sum(gate_weights * expert_outputs, dim=1)  # (batch, embedding_dim)

            # Task-specific prediction
            task_pred = torch.sigmoid(self.towers[i](task_repr))
            predictions.append(task_pred)

        return predictions  # [ctr_pred, cvr_pred]


class MultiTaskLoss(nn.Module):
    """
    Loss function for multi-task learning with learnable task weights.
    """

    def __init__(self, num_tasks=2, loss_type='weighted'):
        super(MultiTaskLoss, self).__init__()

        self.num_tasks = num_tasks
        self.loss_type = loss_type

        if loss_type == 'learnable':
            # Learnable task weights
            self.task_weights = nn.Parameter(torch.ones(num_tasks))

        self.bce = nn.BCELoss()

    def forward(self, predictions, labels, aux_predictions=None):
        """
        Compute multi-task loss.

        Args:
            predictions: List of task predictions
            labels: List of task labels
            aux_predictions: Optional auxiliary predictions

        Returns:
            total_loss: Combined loss
            loss_dict: Individual task losses
        """
        loss_dict = {}

        for i, (pred, label) in enumerate(zip(predictions, labels)):
            task_loss = self.bce(pred.squeeze(), label)
            loss_dict[f'task_{i}_loss'] = task_loss.item()

        if self.loss_type == 'equal':
            total_loss = sum(loss_dict.values()) / self.num_tasks

        elif self.loss_type == 'weighted':
            # Fixed weights based on task difficulty
            weights = [0.4, 0.6]  # CTR usually more reliable
            total_loss = sum(w * l for w, l in zip(weights, loss_dict.values()))

        elif self.loss_type == 'learnable':
            # Learnable weights with softmax normalization
            weights = F.softmax(self.task_weights, dim=0)
            total_loss = sum(w * l for w, l in zip(weights, loss_dict.values()))

        # Add auxiliary loss if provided
        if aux_predictions is not None:
            aux_loss = self.bce(aux_predictions.squeeze(), labels[0])
            total_loss = total_loss + 0.1 * aux_loss
            loss_dict['aux_loss'] = aux_loss.item()

        return total_loss, loss_dict


class MultiTaskTrainer:
    """
    Trainer for multi-task learning models.
    """

    def __init__(self, model, optimizer, device='cuda', loss_type='weighted'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = MultiTaskLoss(loss_type=loss_type)
        self.model.to(device)

    def train_step(self, batch):
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        # Move data to device
        features = batch['features'].to(self.device)
        ctr_labels = batch['ctr_labels'].to(self.device)
        cvr_labels = batch['cvr_labels'].to(self.device)

        # Forward pass
        if isinstance(self.model, MMoE):
            predictions = self.model(features)
            ctr_pred, cvr_pred = predictions[0], predictions[1]
        else:
            ctr_pred, cvr_pred, _ = self.model(features)

        # Compute loss
        predictions = [ctr_pred, cvr_pred]
        labels = [ctr_labels, cvr_labels]

        loss, loss_dict = self.criterion(predictions, labels)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Add predictions to loss dict
        loss_dict['ctr_pred_mean'] = ctr_pred.mean().item()
        loss_dict['cvr_pred_mean'] = cvr_pred.mean().item()

        return loss.item(), loss_dict

    def evaluate(self, dataloader):
        """Evaluate model on validation set."""
        self.model.eval()

        all_ctr_preds = []
        all_cvr_preds = []
        all_ctr_labels = []
        all_cvr_labels = []
        total_loss = 0

        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                ctr_labels = batch['ctr_labels'].to(self.device)
                cvr_labels = batch['cvr_labels'].to(self.device)

                # Forward pass
                if isinstance(self.model, MMoE):
                    predictions = self.model(features)
                    ctr_pred, cvr_pred = predictions[0], predictions[1]
                else:
                    ctr_pred, cvr_pred, _ = self.model(features)

                # Compute loss
                predictions = [ctr_pred, cvr_pred]
                labels = [ctr_labels, cvr_labels]
                loss, _ = self.criterion(predictions, labels)
                total_loss += loss.item()

                all_ctr_preds.extend(ctr_pred.cpu().numpy().flatten())
                all_cvr_preds.extend(cvr_pred.cpu().numpy().flatten())
                all_ctr_labels.extend(ctr_labels.cpu().numpy())
                all_cvr_labels.extend(cvr_labels.cpu().numpy())

        # Compute metrics
        from sklearn.metrics import roc_auc_score, accuracy_score

        results = {
            'loss': total_loss / len(dataloader),
            'ctr_auc': roc_auc_score(all_ctr_labels, all_ctr_preds) if len(set(all_ctr_labels)) > 1 else 0.5,
            'cvr_auc': roc_auc_score(all_cvr_labels, all_cvr_preds) if len(set(all_cvr_labels)) > 1 else 0.5,
            'ctr_accuracy': accuracy_score(all_ctr_labels, [1 if p > 0.5 else 0 for p in all_ctr_preds]),
            'cvr_accuracy': accuracy_score(all_cvr_labels, [1 if p > 0.5 else 0 for p in all_cvr_preds])
        }

        return results