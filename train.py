"""
Main Training Script for Multi-Modal E-Commerce Recommendation System
=====================================================================

This script provides end-to-end training, evaluation, and comparison
of recommendation models for e-commerce scenarios.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import argparse
import os
import json
from datetime import datetime

# Import project modules
from models.dssm import DSSMModel, DSSMTrainer
from models.din import DINModel, DINLoss
from models.multi_task import ESMM, MMoE, MultiTaskTrainer
from data.data_loader import (
    create_synthetic_data, split_data, create_dataloaders, EcommerceDataset
)
from evaluation.ab_testing import RecommendationEvaluator, simulate_ab_test_data, ABTestEvaluator


class TrainingManager:
    """
    Manages training, evaluation, and model selection.
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize data
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # Initialize models
        self.models = {}
        self.trainers = {}

    def setup_data(self):
        """Setup data loaders."""
        print("\n" + "=" * 50)
        print("Setting up data...")
        print("=" * 50)

        # Create synthetic data
        df = create_synthetic_data(
            num_users=self.config.get('num_users', 10000),
            num_items=self.config.get('num_items', 50000),
            num_interactions=self.config.get('num_interactions', 100000),
            max_behavior_seq_len=self.config.get('max_behavior_seq_len', 50)
        )

        # Split data
        train_df, val_df, test_df = split_data(df)

        # Create dataloaders
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            train_df, val_df, test_df, self.config
        )

        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Test batches: {len(self.test_loader)}")

    def setup_models(self):
        """Initialize models and trainers."""
        print("\n" + "=" * 50)
        print("Setting up models...")
        print("=" * 50)

        model_config = {
            'embedding_dim': self.config.get('embedding_dim', 128),
            'hidden_dims': self.config.get('hidden_dims', [256, 128, 64]),
            'dropout': self.config.get('dropout', 0.2),
            'user_feature_dim': self.config.get('user_feature_dim', 64),
            'item_feature_dim': self.config.get('item_feature_dim', 128),
            'use_multimodal': self.config.get('use_multimodal', True)
        }

        # DSSM Model
        dssm_config = {**model_config, 'num_items': self.config.get('num_items', 50000)}
        self.models['dssm'] = DSSMModel(dssm_config).to(self.device)
        self.trainers['dssm'] = DSSMTrainer(
            self.models['dssm'],
            optim.Adam(self.models['dssm'].parameters(), lr=self.config.get('lr', 0.001)),
            device=self.device
        )
        print("DSSM model initialized")

        # DIN Model
        din_config = {
            **model_config,
            'max_behavior_seq_len': self.config.get('max_behavior_seq_len', 50),
            'num_items': self.config.get('num_items', 50000)
        }
        self.models['din'] = DINModel(din_config).to(self.device)
        self.trainers['din'] = DINLoss(auxiliary_weight=0.2)
        print("DIN model initialized")

        # Multi-task Models
        multi_task_config = {
            'embedding_dim': self.config.get('embedding_dim', 128),
            'input_dim': self.config.get('user_feature_dim', 64) + self.config.get('embedding_dim', 128) * 3,
            'dropout': self.config.get('dropout', 0.2)
        }

        self.models['esmm'] = ESMM(multi_task_config).to(self.device)
        self.trainers['esmm'] = MultiTaskTrainer(
            self.models['esmm'],
            optim.Adam(self.models['esmm'].parameters(), lr=self.config.get('lr', 0.001)),
            device=self.device
        )
        print("ESMM model initialized")

        self.models['mmoe'] = MMoE({
            **multi_task_config,
            'num_experts': 4,
            'num_tasks': 2
        }).to(self.device)
        self.trainers['mmoe'] = MultiTaskTrainer(
            self.models['mmoe'],
            optim.Adam(self.models['mmoe'].parameters(), lr=self.config.get('lr', 0.001)),
            device=self.device,
            loss_type='learnable'
        )
        print("MMoE model initialized")

    def train_model(self, model_name: str, epochs: int = None):
        """Train a specific model."""
        if epochs is None:
            epochs = self.config.get('epochs', 10)

        print(f"\n{'=' * 50}")
        print(f"Training {model_name.upper()} model...")
        print(f"{'=' * 50}")

        model = self.models[model_name]
        trainer = self.trainers[model_name]
        best_val_auc = 0.0

        for epoch in range(epochs):
            # Training
            model.train()
            epoch_losses = []

            for batch in self.train_loader:
                if model_name == 'dssm':
                    loss = trainer.train_step(batch)
                elif model_name == 'din':
                    # DIN training step
                    user_features = batch['user_features'].to(self.device)
                    target_item_ids = batch['item_id'] if isinstance(batch['item_id'], torch.Tensor) else torch.tensor(batch['item_id']).to(self.device)
                    behavior_seq = batch['behavior_seq'].to(self.device)
                    behavior_length = batch['behavior_length'].to(self.device)
                    labels = batch['label'].to(self.device)

                    ctr_score, _ = model(user_features, target_item_ids, behavior_seq, behavior_length)
                    loss, _ = trainer.criterion(ctr_score, labels)
                    trainer.optimizer.zero_grad()
                    loss.backward()
                    trainer.optimizer.step()
                elif model_name in ['esmm', 'mmoe']:
                    loss, _ = trainer.train_step(batch)
                else:
                    loss = 0

                epoch_losses.append(loss)

            avg_loss = sum(epoch_losses) / len(epoch_losses)

            # Validation
            if self.val_loader is not None:
                val_results = self.evaluate_model(model_name)
                val_auc = val_results.get('auc', 0.0)

                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val AUC: {val_auc:.4f}")

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    self.save_model(model_name, epoch, val_auc)
            else:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        return best_val_auc

    def evaluate_model(self, model_name: str):
        """Evaluate a model on the test set."""
        model = self.models[model_name]
        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.test_loader:
                user_features = batch['user_features'].to(self.device)
                item_features = batch['item_features'].to(self.device)
                labels = batch['label'].to(self.device)

                if model_name == 'dssm':
                    similarity, _, _ = model(user_features, item_features)
                    probs = torch.sigmoid(similarity)
                else:
                    probs = torch.rand(len(labels)).to(self.device)  # Placeholder

                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute metrics
        from sklearn.metrics import roc_auc_score, accuracy_score

        results = {
            'auc': roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5,
            'accuracy': accuracy_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds])
        }

        return results

    def compare_models(self):
        """Compare all models and generate report."""
        print("\n" + "=" * 50)
        print("Model Comparison")
        print("=" * 50)

        results = {}
        for model_name in self.models.keys():
            results[model_name] = self.evaluate_model(model_name)
            print(f"{model_name.upper()}: AUC={results[model_name]['auc']:.4f}, Accuracy={results[model_name]['accuracy']:.4f}")

        return results

    def save_model(self, model_name: str, epoch: int, val_auc: float):
        """Save model checkpoint."""
        save_dir = os.path.join(self.config.get('save_dir', 'checkpoints'), model_name)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f'model_epoch{epoch}_auc{val_auc:.4f}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.models[model_name].state_dict(),
            'val_auc': val_auc
        }, save_path)
        print(f"Model saved to {save_path}")

    def run_ab_test_simulation(self):
        """Run simulated A/B test."""
        print("\n" + "=" * 50)
        print("Running A/B Test Simulation")
        print("=" * 50)

        # Simulate A/B test data
        assignments, tracker = simulate_ab_test_data(num_users=2000)

        # Evaluate
        evaluator = ABTestEvaluator(
            name="Recommendation_Model_Comparison",
            control_model_name="baseline",
            treatment_model_name="multimodal_din"
        )
        evaluator.tracker = tracker

        report = evaluator.evaluate(metrics=['ctr', 'cvr', 'gmv'])

        print(f"\nA/B Test Results:")
        for metric, res in report['results'].items():
            print(f"  {metric}: Control={res['control_mean']:.4f}, Treatment={res['treatment_mean']:.4f}, Lift={res['lift']*100:+.1f}%")

        return report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Multi-Modal Recommendation Models')

    parser.add_argument('--model', type=str, default='all',
                        choices=['dssm', 'din', 'esmm', 'mmoe', 'all'],
                        help='Model to train')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Embedding dimension')
    parser.add_argument('--num_users', type=int, default=10000,
                        help='Number of users in synthetic data')
    parser.add_argument('--num_items', type=int, default=50000,
                        help='Number of items in synthetic data')
    parser.add_argument('--num_interactions', type=int, default=100000,
                        help='Number of interactions in synthetic data')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save models')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config JSON file')

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Load config from file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Use command line arguments
        config = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'embedding_dim': args.embedding_dim,
            'num_users': args.num_users,
            'num_items': args.num_items,
            'num_interactions': args.num_interactions,
            'save_dir': args.save_dir,
            'max_behavior_seq_len': 50,
            'user_feature_dim': 64,
            'item_feature_dim': 128,
            'hidden_dims': [256, 128, 64],
            'dropout': 0.2,
            'use_multimodal': True
        }

    # Initialize training manager
    manager = TrainingManager(config)

    # Setup
    manager.setup_data()
    manager.setup_models()

    # Train selected model(s)
    if args.model == 'all':
        for model_name in ['dssm', 'din', 'esmm', 'mmoe']:
            manager.train_model(model_name)
    else:
        manager.train_model(args.model)

    # Compare models
    results = manager.compare_models()

    # Run A/B test simulation
    ab_report = manager.run_ab_test_simulation()

    # Save final report
    report_path = os.path.join(config['save_dir'], 'training_report.json')
    os.makedirs(config['save_dir'], exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'model_results': results,
            'ab_test_results': ab_report
        }, f, indent=2)

    print(f"\nTraining complete! Report saved to {report_path}")


if __name__ == '__main__':
    main()