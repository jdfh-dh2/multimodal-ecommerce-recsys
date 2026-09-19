"""
Deep Interest Network (DIN) Implementation
===========================================

This module implements the DIN attention mechanism for modeling
user behavior sequences in e-commerce recommendations.

Reference: Zhou et al. "Deep Interest Evolution Network for Click-Through Rate Prediction"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentionLayer(nn.Module):
    """
    Attention mechanism for user behavior sequence.
    Implements the Dice activation function and attention pooling.
    """

    def __init__(self, embedding_dim, attention_units=[36, 8], dice_dim=36):
        super(AttentionLayer, self).__init__()

        self.embedding_dim = embedding_dim
        self.dice_dim = dice_dim

        # Query, Key, Value projection
        self.query_projection = nn.Linear(embedding_dim, attention_units[1])
        self.key_projection = nn.Linear(embedding_dim, attention_units[1])
        self.value_projection = nn.Linear(embedding_dim, attention_units[1])

        # Dice activation batch norm
        self.bn = nn.BatchNorm1d(attention_units[1])

        # Output projection
        self.output_projection = nn.Linear(attention_units[1], embedding_dim)

    def forward(self, query, keys, key_length=None):
        """
        Compute attention-weighted sum of values.

        Args:
            query: Current item embedding (batch_size, embedding_dim)
            keys: Behavior sequence embeddings (batch_size, seq_len, embedding_dim)
            key_length: Actual sequence length (batch_size,)

        Returns:
            output: Attention-weighted representation (batch_size, embedding_dim)
            attention_weights: Attention scores (batch_size, seq_len)
        """
        batch_size, seq_len, _ = keys.size()

        # Expand query for sequence
        query_expanded = query.unsqueeze(1).expand_as(keys)  # (batch, seq_len, dim)

        # Concatenate query and keys for attention input
        attention_input = torch.cat([query_expanded, keys, query_expanded - keys], dim=-1)

        # Project to attention space
        queries = self.query_projection(query)  # (batch, attention_dim)
        keys_transformed = self.key_projection(keys.view(-1, self.embedding_dim))  # (batch*seq, attention_dim)
        keys_transformed = keys_transformed.view(batch_size, seq_len, -1)  # (batch, seq, attention_dim)

        # Compute attention scores
        attention_scores = torch.sum(queries.unsqueeze(1) * keys_transformed, dim=-1)  # (batch, seq)
        attention_scores = attention_scores / math.sqrt(self.embedding_dim)

        # Mask padding positions
        if key_length is not None:
            mask = torch.arange(seq_len, device=keys.device).unsqueeze(0) < key_length.unsqueeze(1)
            attention_scores = attention_scores.masked_fill(~mask, float('-inf'))

        # Softmax to get weights
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, seq)

        # Weighted sum of values
        values = self.value_projection(keys.view(-1, self.embedding_dim))
        values = values.view(batch_size, seq_len, -1)  # (batch, seq, attention_dim)

        output = torch.sum(attention_weights.unsqueeze(-1) * values, dim=1)  # (batch, attention_dim)

        # Output projection with residual
        output = self.output_projection(output)
        output = output + query  # Residual connection

        return output, attention_weights


class Dice(nn.Module):
    """
    Dice activation function for automatic regularization.
    """

    def __init__(self, dim=-1):
        super(Dice, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm1d(1)

    def forward(self, x):
        # Sigmoid activation
        p = torch.sigmoid(x)

        # Dice normalization
        mean = x.mean(dim=self.dim, keepdim=True)
        var = x.var(dim=self.dim, keepdim=True)
        normalized_x = (x - mean) / torch.sqrt(var + 1e-8)

        # Dice activation
        output = p * x + (1 - p) * normalized_x

        return output


class DINModel(nn.Module):
    """
    Deep Interest Network for CTR prediction.

    Uses attention mechanism to dynamically learn the relevance
    of each behavior in the user's history to the current target item.
    """

    def __init__(self, config):
        super(DINModel, self).__init__()

        self.config = config
        self.embedding_dim = config.get('embedding_dim', 128)
        self.behavior_seq_len = config.get('max_behavior_seq_len', 50)
        self.hidden_dims = config.get('hidden_dims', [200, 80])

        # Behavior sequence encoder
        self.behavior_encoder = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.embedding_dim,
            batch_first=True,
            bidirectional=True
        )

        # Attention layer for behavior sequence
        self.attention = AttentionLayer(
            embedding_dim=self.embedding_dim,
            attention_units=[36, 8]
        )

        # User profile projection
        self.user_projection = nn.Sequential(
            nn.Linear(config.get('user_feature_dim', 64), self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.2))
        )

        # Item embedding
        self.item_embedding = nn.Embedding(
            num_embeddings=config.get('num_items', 10000),
            embedding_dim=self.embedding_dim
        )

        # Multi-modal item features
        self.use_multimodal = config.get('use_multimodal', True)
        if self.use_multimodal:
            self.image_projection = nn.Linear(2048, self.embedding_dim)
            self.text_projection = nn.Linear(768, self.embedding_dim)

        # MLP layers for CTR prediction
        mlp_input_dim = self.embedding_dim * 4  # user + target_item + attended_behavior + user_item_interaction
        if self.use_multimodal:
            mlp_input_dim += self.embedding_dim * 2  # image + text features

        layers = []
        prev_dim = mlp_input_dim
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                Dice(),
                nn.Dropout(config.get('dropout', 0.2))
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_features, target_item_ids, behavior_seq, behavior_length, item_image_features=None, item_text_features=None):
        """
        Forward pass for CTR prediction.

        Args:
            user_features: User profile features (batch_size, user_feature_dim)
            target_item_ids: Target item IDs (batch_size,)
            behavior_seq: User behavior history (batch_size, seq_len)
            behavior_length: Actual behavior sequence length (batch_size,)
            item_image_features: Optional image features (batch_size, 2048)
            item_text_features: Optional text features (batch_size, 768)

        Returns:
            ctr_score: Predicted CTR (batch_size, 1)
            attention_weights: Attention scores for visualization
        """
        # User representation
        user_emb = self.user_projection(user_features)  # (batch, embedding_dim)

        # Target item embedding
        target_item_emb = self.item_embedding(target_item_ids)  # (batch, embedding_dim)

        # Multi-modal features for target item
        if self.use_multimodal and item_image_features is not None and item_text_features is not None:
            img_emb = self.image_projection(item_image_features)
            text_emb = self.text_projection(item_text_features)
            target_item_emb = target_item_emb + img_emb + text_emb

        # Behavior sequence encoding with attention
        batch_size, seq_len, _ = behavior_seq.size()

        # Encode behavior sequence with GRU
        behavior_encoded, _ = self.behavior_encoder(behavior_seq)  # (batch, seq, hidden*2)

        # Apply attention mechanism
        attended_behavior, attention_weights = self.attention(target_item_emb, behavior_encoded, behavior_length)

        # Concatenate all features
        interaction = user_emb * target_item_emb  # Element-wise product
        mlp_input = torch.cat([
            user_emb,
            target_item_emb,
            attended_behavior,
            interaction
        ], dim=-1)

        # MLP for CTR prediction
        ctr_score = self.mlp(mlp_input)
        ctr_score = torch.sigmoid(ctr_score)

        return ctr_score, attention_weights

    def get_attention_visualization(self, user_features, target_item_ids, behavior_seq, behavior_length):
        """
        Get attention weights for visualization.
        Useful for debugging and understanding model behavior.
        """
        with torch.no_grad():
            user_emb = self.user_projection(user_features)
            target_item_emb = self.item_embedding(target_item_ids)

            batch_size, seq_len, _ = behavior_seq.size()
            behavior_encoded, _ = self.behavior_encoder(behavior_seq)

            _, attention_weights = self.attention(target_item_emb, behavior_encoded, behavior_length)

        return attention_weights


class DINLoss(nn.Module):
    """
    Loss function for DIN model with auxiliary loss for sequence encoding.
    """

    def __init__(self, auxiliary_weight=0.2):
        super(DINLoss, self).__init__()
        self.auxiliary_weight = auxiliary_weight
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, predictions, labels, auxiliary_predictions=None, auxiliary_labels=None):
        """
        Compute loss with optional auxiliary loss.

        Args:
            predictions: Main CTR predictions
            labels: Ground truth labels
            auxiliary_predictions: Optional auxiliary predictions (next item)
            auxiliary_labels: Optional auxiliary labels
        """
        # Main loss
        main_loss = self.bce(predictions.squeeze(), labels).mean()

        total_loss = main_loss

        # Auxiliary loss for better sequence representation
        if auxiliary_predictions is not None and auxiliary_labels is not None:
            aux_loss = self.bce(auxiliary_predictions.squeeze(), auxiliary_labels).mean()
            total_loss = total_loss + self.auxiliary_weight * aux_loss

        return total_loss, main_loss.item()


class DIN Trainer
    """
    Trainer class for DIN model.
    """

    def __init__(self, model, optimizer, device='cuda', auxiliary_weight=0.2):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = DINLoss(auxiliary_weight=auxiliary_weight)
        self.model.to(device)

    def train_step(self, batch):
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        # Move data to device
        user_features = batch['user_features'].to(self.device)
        target_item_ids = batch['target_item_ids'].to(self.device)
        behavior_seq = batch['behavior_seq'].to(self.device)
        behavior_length = batch['behavior_length'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Forward pass
        ctr_score, attention_weights = self.model(
            user_features, target_item_ids, behavior_seq, behavior_length
        )

        # Compute loss
        loss, main_loss = self.criterion(ctr_score, labels)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'main_loss': main_loss,
            'ctr_score': ctr_score.mean().item()
        }

    def evaluate(self, dataloader):
        """Evaluate model on validation set."""
        self.model.eval()

        all_preds = []
        all_labels = []
        total_loss = 0

        with torch.no_grad():
            for batch in dataloader:
                user_features = batch['user_features'].to(self.device)
                target_item_ids = batch['target_item_ids'].to(self.device)
                behavior_seq = batch['behavior_seq'].to(self.device)
                behavior_length = batch['behavior_length'].to(self.device)
                labels = batch['labels'].to(self.device)

                ctr_score, _ = self.model(
                    user_features, target_item_ids, behavior_seq, behavior_length
                )

                loss, _ = self.criterion(ctr_score, labels)
                total_loss += loss.item()

                all_preds.extend(ctr_score.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy())

        # Compute metrics
        from sklearn.metrics import roc_auc_score, accuracy_score

        results = {
            'loss': total_loss / len(dataloader),
            'auc': roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5,
            'accuracy': accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
        }

        return results