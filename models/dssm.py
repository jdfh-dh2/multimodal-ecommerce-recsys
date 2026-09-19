"""
Dual-Tower Model (DSSM) for E-Commerce Recommendation
======================================================

This module implements the Deep Structured Semantic Model (DSSM) architecture
for learning semantic representations of users and items in e-commerce.

Reference: Huang et al. "Learning Deep Structured Semantic Models for Web Search"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSSMModel(nn.Module):
    """
    Dual-Tower Model for E-Commerce Recommendation.

    Uses separate towers for user and item representations,
    then computes similarity in a shared latent space.
    """

    def __init__(self, config):
        super(DSSMModel, self).__init__()

        self.config = config
        self.embedding_dim = config.get('embedding_dim', 128)
        self.hidden_dims = config.get('hidden_dims', [256, 128, 64])
        self.dropout = config.get('dropout', 0.2)

        # User tower
        self.user_tower = self._build_tower(
            input_dim=config.get('user_feature_dim', 64),
            hidden_dims=self.hidden_dims,
            output_dim=self.embedding_dim
        )

        # Item tower
        self.item_tower = self._build_tower(
            input_dim=config.get('item_feature_dim', 128),
            hidden_dims=self.hidden_dims,
            output_dim=self.embedding_dim
        )

        # Multi-modal fusion layer
        self.use_multimodal = config.get('use_multimodal', True)
        if self.use_multimodal:
            self.image_projection = nn.Linear(2048, 256)
            self.text_projection = nn.Linear(768, 256)

    def _build_tower(self, input_dim, hidden_dims, output_dim):
        """Build a tower network."""
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

    def forward_user(self, user_features):
        """
        Forward pass for user representation.

        Args:
            user_features: Tensor of shape (batch_size, user_feature_dim)
        Returns:
            user_embedding: Tensor of shape (batch_size, embedding_dim)
        """
        return self.user_tower(user_features)

    def forward_item(self, item_features):
        """
        Forward pass for item representation.

        Args:
            item_features: Tensor of shape (batch_size, item_feature_dim)
        Returns:
            item_embedding: Tensor of shape (batch_size, embedding_dim)
        """
        return self.item_tower(item_features)

    def forward(self, user_features, item_features, user_seq_features=None, item_image_features=None, item_text_features=None):
        """
        Full forward pass.

        Args:
            user_features: User feature tensor
            item_features: Item feature tensor
            user_seq_features: Optional user behavior sequence features
            item_image_features: Optional image features for multi-modal
            item_text_features: Optional text features for multi-modal
        """
        # User representation
        user_emb = self.forward_user(user_features)

        # Item representation (with optional multi-modal fusion)
        if self.use_multimodal and item_image_features is not None and item_text_features is not None:
            # Project multi-modal features
            img_proj = self.image_projection(item_image_features)
            text_proj = self.text_projection(item_text_features)
            # Concatenate with base item features
            fused_item_features = torch.cat([item_features, img_proj, text_proj], dim=-1)
            item_emb = self.forward_item(fused_item_features)
        else:
            item_emb = self.forward_item(item_features)

        # Compute cosine similarity
        user_emb = F.normalize(user_emb, p=2, dim=1)
        item_emb = F.normalize(item_emb, p=2, dim=1)
        similarity = torch.sum(user_emb * item_emb, dim=1)

        return similarity, user_emb, item_emb

    def compute_loss(self, similarity, labels):
        """
        Compute cross-entropy loss for ranking.

        Args:
            similarity: Predicted similarity scores
            labels: Ground truth labels (1 for positive, 0 for negative)
        """
        # Convert similarity to probabilities using sigmoid
        probs = torch.sigmoid(similarity)
        loss = F.binary_cross_entropy(probs, labels)
        return loss


class MultiModalFusion(nn.Module):
    """
    Multi-modal fusion module combining image and text features.
    """

    def __init__(self, image_dim=2048, text_dim=768, fusion_dim=256):
        super(MultiModalFusion, self).__init__()

        # Image processing
        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )

        # Text processing
        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=4,
            dropout=0.1
        )

        # Output fusion
        self.output_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, image_features, text_features):
        """
        Fuse image and text features.

        Args:
            image_features: Image embeddings (batch_size, image_dim)
            text_features: Text embeddings (batch_size, text_dim)
        Returns:
            fused_features: Combined multi-modal features (batch_size, fusion_dim)
        """
        # Project to common space
        img_proj = self.image_projection(image_features)
        text_proj = self.text_projection(text_features)

        # Cross-modal attention
        img_proj = img_proj.unsqueeze(1)  # (batch, 1, dim)
        text_proj = text_proj.unsqueeze(1)  # (batch, 1, dim)

        attended_img, _ = self.cross_attention(img_proj, text_proj, text_proj)
        attended_text, _ = self.cross_attention(text_proj, img_proj, img_proj)

        attended_img = attended_img.squeeze(1)  # (batch, dim)
        attended_text = attended_text.squeeze(1)  # (batch, dim)

        # Concatenate and fuse
        fused = torch.cat([attended_img, attended_text], dim=-1)
        output = self.output_fusion(fused)

        return output


def create_negative_samples(item_embeddings, num_negative=4):
    """
    Create negative samples for training.

    Args:
        item_embeddings: Item embeddings (num_items, embedding_dim)
        num_negative: Number of negative samples per positive
    Returns:
        negative_indices: Indices of negative samples
    """
    num_items = item_embeddings.size(0)
    negative_indices = torch.randint(0, num_items, (num_negative,))
    return negative_indices


class DSSMTrainer:
    """
    Trainer for DSSM model.
    """

    def __init__(self, model, optimizer, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)

    def train_step(self, batch):
        """
        Single training step.

        Args:
            batch: Dictionary with 'user_features', 'item_features', 'labels'
        """
        self.model.train()
        self.optimizer.zero_grad()

        user_features = batch['user_features'].to(self.device)
        item_features = batch['item_features'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Forward pass
        similarity, _, _ = self.model(user_features, item_features)

        # Compute loss
        loss = self.model.compute_loss(similarity, labels)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, dataloader, metrics=['auc', 'accuracy', 'mrr']):
        """
        Evaluate model on validation set.
        """
        self.model.eval()

        all_preds = []
        all_labels = []
        total_loss = 0

        with torch.no_grad():
            for batch in dataloader:
                user_features = batch['user_features'].to(self.device)
                item_features = batch['item_features'].to(self.device)
                labels = batch['labels'].to(self.device)

                similarity, _, _ = self.model(user_features, item_features)
                loss = self.model.compute_loss(similarity, labels)
                total_loss += loss.item()

                probs = torch.sigmoid(similarity)
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute metrics
        results = {}
        results['loss'] = total_loss / len(dataloader)

        all_preds = torch.tensor(all_preds)
        all_labels = torch.tensor(all_labels)

        if 'auc' in metrics:
            results['auc'] = compute_auc(all_labels, all_preds)
        if 'accuracy' in metrics:
            results['accuracy'] = compute_accuracy(all_labels, all_preds)
        if 'mrr' in metrics:
            results['mrr'] = compute_mrr(all_labels, all_preds)

        return results


def compute_auc(labels, predictions):
    """Compute AUC score."""
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(labels, predictions)
    except:
        return 0.0


def compute_accuracy(labels, predictions, threshold=0.5):
    """Compute accuracy."""
    preds_binary = (predictions >= threshold).float()
    return (preds_binary == labels).float().mean().item()


def compute_mrr(labels, predictions, k=10):
    """
    Compute Mean Reciprocal Rank.

    For each user, find the rank of the first relevant item.
    """
    # Simplified MRR computation
    preds_sorted = torch.argsort(predictions, descending=True)
    labels_sorted = labels[preds_sorted]

    # Find first relevant item
    relevant_ranks = (labels_sorted == 1).nonzero(as_tuple=True)[0]
    if len(relevant_ranks) == 0:
        return 0.0

    first_rank = relevant_ranks[0].item() + 1  # 1-indexed
    return 1.0 / first_rank