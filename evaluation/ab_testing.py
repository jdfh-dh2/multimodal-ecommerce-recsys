"""
A/B Testing Framework for Recommendation Systems
=================================================

This module implements a comprehensive A/B testing framework for
evaluating recommendation models in production scenarios.
"""

import numpy as np
import torch
from scipy import stats
from typing import Dict, List, Tuple, Optional
import json
import hashlib
import random


class ABTestSplitter:
    """
    Split users into control and treatment groups for A/B testing.
    """

    def __init__(self, seed=42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.manual_seed(seed)

    def assign_user(self, user_id: str, treatment_ratio: float = 0.5) -> str:
        """
        Assign a user to a group based on their ID hash.

        Args:
            user_id: Unique user identifier
            treatment_ratio: Ratio of users to put in treatment group

        Returns:
            'treatment' or 'control'
        """
        # Hash user ID for consistent assignment
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        group = 'treatment' if (hash_value % 100) < (treatment_ratio * 100) else 'control'
        return group

    def assign_batch(self, user_ids: List[str], treatment_ratio: float = 0.5) -> Dict[str, str]:
        """
        Assign a batch of users to groups.

        Args:
            user_ids: List of user IDs
            treatment_ratio: Ratio of users in treatment

        Returns:
            Dictionary mapping user_id to group
        """
        return {user_id: self.assign_user(user_id, treatment_ratio) for user_id in user_ids}


class MetricsTracker:
    """
    Track metrics for A/B test groups.
    """

    def __init__(self):
        self.control_metrics = {
            'ctr': [], 'cvr': [], 'gmv': [], 'session_length': [],
            'items_viewed': [], 'items_purchased': [], 'user_rating': []
        }
        self.treatment_metrics = {
            'ctr': [], 'cvr': [], 'gmv': [], 'session_length': [],
            'items_viewed': [], 'items_purchased': [], 'user_rating': []
        }

    def record(self, group: str, metrics: Dict[str, float]):
        """
        Record metrics for a user session.

        Args:
            group: 'control' or 'treatment'
            metrics: Dictionary of metric values
        """
        target = self.control_metrics if group == 'control' else self.treatment_metrics

        for key, value in metrics.items():
            if key in target:
                target[key].append(value)

    def get_summary(self, group: str) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for a group.

        Returns:
            Dictionary with mean, std, count for each metric
        """
        target = self.control_metrics if group == 'control' else self.treatment_metrics

        summary = {}
        for metric_name, values in target.items():
            if len(values) > 0:
                summary[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }
            else:
                summary[metric_name] = {
                    'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0
                }

        return summary

    def reset(self):
        """Reset all tracked metrics."""
        for key in self.control_metrics:
            self.control_metrics[key] = []
            self.treatment_metrics[key] = []


class StatisticalTest:
    """
    Perform statistical significance tests for A/B test results.
    """

    @staticmethod
    def t_test(control: List[float], treatment: List[float],
               alpha: float = 0.05) -> Dict[str, any]:
        """
        Perform independent t-test.

        Args:
            control: Control group values
            treatment: Treatment group values
            alpha: Significance level

        Returns:
            Dictionary with test results
        """
        if len(control) < 2 or len(treatment) < 2:
            return {
                'significant': False,
                'p_value': 1.0,
                't_statistic': 0.0,
                'alpha': alpha,
                'error': 'Insufficient samples'
            }

        t_stat, p_value = stats.ttest_ind(control, treatment)

        return {
            'significant': p_value < alpha,
            'p_value': float(p_value),
            't_statistic': float(t_stat),
            'alpha': alpha,
            'control_mean': float(np.mean(control)),
            'treatment_mean': float(np.mean(treatment)),
            'lift': float((np.mean(treatment) - np.mean(control)) / (np.mean(control) + 1e-8))
        }

    @staticmethod
    def mann_whitney_u(control: List[float], treatment: List[float],
                       alpha: float = 0.05) -> Dict[str, any]:
        """
        Perform Mann-Whitney U test (non-parametric).

        Args:
            control: Control group values
            treatment: Treatment group values
            alpha: Significance level

        Returns:
            Dictionary with test results
        """
        if len(control) < 2 or len(treatment) < 2:
            return {
                'significant': False,
                'p_value': 1.0,
                'u_statistic': 0.0,
                'alpha': alpha,
                'error': 'Insufficient samples'
            }

        statistic, p_value = stats.mannwhitneyu(control, treatment, alternative='two-sided')

        return {
            'significant': p_value < alpha,
            'p_value': float(p_value),
            'u_statistic': float(statistic),
            'alpha': alpha,
            'control_median': float(np.median(control)),
            'treatment_median': float(np.median(treatment))
        }

    @staticmethod
    def chi_square_test(control_success: int, control_total: int,
                        treatment_success: int, treatment_total: int,
                        alpha: float = 0.05) -> Dict[str, any]:
        """
        Perform chi-square test for proportions (e.g., CTR).

        Args:
            control_success: Number of successes in control
            control_total: Total samples in control
            treatment_success: Number of successes in treatment
            treatment_total: Total samples in treatment

        Returns:
            Dictionary with test results
        """
        # Create contingency table
        contingency = np.array([
            [control_success, control_total - control_success],
            [treatment_success, treatment_total - treatment_success]
        ])

        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

        # Calculate proportions
        control_rate = control_success / control_total if control_total > 0 else 0
        treatment_rate = treatment_success / treatment_total if treatment_total > 0 else 0

        return {
            'significant': p_value < alpha,
            'p_value': float(p_value),
            'chi2_statistic': float(chi2),
            'dof': int(dof),
            'alpha': alpha,
            'control_rate': float(control_rate),
            'treatment_rate': float(treatment_rate),
            'lift': float((treatment_rate - control_rate) / (control_rate + 1e-8))
        }

    @staticmethod
    def sequential_test(control_values: List[float], treatment_values: List[float],
                        metric_name: str, max_samples: int = 10000) -> Dict[str, any]:
        """
        Perform sequential test for early stopping.

        Args:
            control_values: Control group values
            treatment_values: Treatment group values
            metric_name: Name of the metric being tested
            max_samples: Maximum samples to test

        Returns:
            Dictionary with sequential test results
        """
        results = {
            'can_stop': False,
            'winner': None,
            'current_samples': len(control_values),
            'confidence': 0.0
        }

        if len(control_values) < 100 or len(treatment_values) < 100:
            return results

        # Compute current means
        control_mean = np.mean(control_values[-max_samples:])
        treatment_mean = np.mean(treatment_values[-max_samples:])

        # Compute confidence interval
        control_std = np.std(control_values[-max_samples:]) / np.sqrt(len(control_values[-max_samples:]))
        treatment_std = np.std(treatment_values[-max_samples:]) / np.sqrt(len(treatment_values[-max_samples:]))

        diff = abs(treatment_mean - control_mean)
        se = np.sqrt(control_std**2 + treatment_std**2)

        if se > 0:
            z_score = diff / se
            confidence = stats.norm.cdf(z_score) * 2 - 1  # Convert to 0-1 scale

            results['confidence'] = float(confidence)
            results['z_score'] = float(z_score)

            # Early stopping criteria
            if confidence > 0.95 and diff > 0.01 * control_mean:
                results['can_stop'] = True
                results['winner'] = 'treatment' if treatment_mean > control_mean else 'control'

        return results


class ABTestEvaluator:
    """
    Main A/B test evaluation class.
    """

    def __init__(self, name: str, control_model_name: str, treatment_model_name: str):
        self.name = name
        self.control_model_name = control_model_name
        self.treatment_model_name = treatment_model_name
        self.splitter = ABTestSplitter()
        self.tracker = MetricsTracker()

    def evaluate(self, metrics: List[str] = None) -> Dict[str, any]:
        """
        Perform comprehensive A/B test evaluation.

        Args:
            metrics: List of metrics to evaluate (default: all)

        Returns:
            Evaluation report
        """
        if metrics is None:
            metrics = ['ctr', 'cvr', 'gmv', 'session_length']

        report = {
            'test_name': self.name,
            'control_model': self.control_model_name,
            'treatment_model': self.treatment_model_name,
            'sample_sizes': {},
            'results': {},
            'recommendations': []
        }

        # Get sample sizes
        control_summary = self.tracker.get_summary('control')
        treatment_summary = self.tracker.get_summary('treatment')

        report['sample_sizes'] = {
            'control': control_summary['ctr']['count'],
            'treatment': treatment_summary['ctr']['count']
        }

        # Evaluate each metric
        for metric in metrics:
            control_values = self.tracker.control_metrics.get(metric, [])
            treatment_values = self.tracker.treatment_metrics.get(metric, [])

            if len(control_values) > 0 and len(treatment_values) > 0:
                # T-test
                t_result = StatisticalTest.t_test(control_values, treatment_values)

                # Mann-Whitney (non-parametric)
                mw_result = StatisticalTest.mann_whitney_u(control_values, treatment_values)

                report['results'][metric] = {
                    'control_mean': t_result.get('control_mean', 0),
                    'treatment_mean': t_result.get('treatment_mean', 0),
                    'lift': t_result.get('lift', 0),
                    'p_value_ttest': t_result['p_value'],
                    'significant_ttest': t_result['significant'],
                    'p_value_mw': mw_result['p_value'],
                    'significant_mw': mw_result['significant']
                }

                # Add recommendation
                if t_result['significant']:
                    if t_result['lift'] > 0.05:
                        report['recommendations'].append(
                            f"{metric}: Treatment significantly better (+{t_result['lift']*100:.1f}%)"
                        )
                    elif t_result['lift'] < -0.05:
                        report['recommendations'].append(
                            f"{metric}: Control significantly better ({t_result['lift']*100:.1f}%)"
                        )
                else:
                    report['recommendations'].append(
                        f"{metric}: No significant difference detected"
                    )

        return report

    def save_report(self, filepath: str):
        """Save evaluation report to JSON file."""
        report = self.evaluate()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {filepath}")


class RecommendationEvaluator:
    """
    Offline evaluation for recommendation models.
    """

    @staticmethod
    def compute_ctr_metrics(predictions: np.ndarray, labels: np.ndarray,
                            k_values: List[int] = None) -> Dict[str, float]:
        """
        Compute CTR-related metrics.

        Args:
            predictions: Predicted scores
            labels: Ground truth labels
            k_values: Top-K values to evaluate

        Returns:
            Dictionary of metrics
        """
        if k_values is None:
            k_values = [1, 3, 5, 10, 20]

        metrics = {}

        # Sort by predictions
        sorted_indices = np.argsort(-predictions)
        sorted_labels = labels[sorted_indices]

        # Compute Top-K metrics
        for k in k_values:
            top_k_labels = sorted_labels[:k]
            metrics[f'precision@{k}'] = float(np.mean(top_k_labels))
            metrics[f'recall@{k}'] = float(np.sum(top_k_labels) / (np.sum(labels) + 1e-8))

        # MRR (Mean Reciprocal Rank)
        first_relevant = np.where(sorted_labels == 1)[0]
        if len(first_relevant) > 0:
            metrics['mrr'] = float(1.0 / (first_relevant[0] + 1))
        else:
            metrics['mrr'] = 0.0

        # NDCG
        dcg = np.sum((2**sorted_labels - 1) / np.log2(np.arange(len(sorted_labels)) + 2))
        ideal_sorted = np.sort(-labels)
        idcg = np.sum((2**ideal_sorted - 1) / np.log2(np.arange(len(ideal_sorted)) + 2))
        metrics['ndcg'] = float(dcg / (idcg + 1e-8))

        # AUC
        from sklearn.metrics import roc_auc_score
        if len(set(labels)) > 1:
            metrics['auc'] = float(roc_auc_score(labels, predictions))
        else:
            metrics['auc'] = 0.5

        return metrics

    @staticmethod
    def compute_cvr_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Compute CVR-related metrics.

        Args:
            predictions: Predicted CVR scores
            labels: Ground truth labels

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

        metrics = {}

        # AUC
        if len(set(labels)) > 1:
            metrics['auc'] = float(roc_auc_score(labels, predictions))
        else:
            metrics['auc'] = 0.5

        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(labels, predictions)
        metrics['pr_auc'] = float(auc(recall, precision))

        # MSE and MAE
        metrics['mse'] = float(np.mean((predictions - labels) ** 2))
        metrics['mae'] = float(np.mean(np.abs(predictions - labels)))

        return metrics

    @staticmethod
    def compute_multitask_metrics(ctr_preds: np.ndarray, cvr_preds: np.ndarray,
                                  ctr_labels: np.ndarray, cvr_labels: np.ndarray) -> Dict[str, float]:
        """
        Compute metrics for multi-task learning.

        Args:
            ctr_preds: CTR predictions
            cvr_preds: CVR predictions
            ctr_labels: CTR ground truth
            cvr_labels: CVR ground truth

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import roc_auc_score

        metrics = {}

        # Individual task metrics
        if len(set(ctr_labels)) > 1:
            metrics['ctr_auc'] = float(roc_auc_score(ctr_labels, ctr_preds))
        else:
            metrics['ctr_auc'] = 0.5

        if len(set(cvr_labels)) > 1:
            metrics['cvr_auc'] = float(roc_auc_score(cvr_labels, cvr_preds))
        else:
            metrics['cvr_auc'] = 0.5

        # CTCVR = CTR * CVR
        ctcvr_preds = ctr_preds * cvr_preds
        # For CTCVR, we need users who clicked AND converted
        ctcvr_labels = ctr_labels * cvr_labels  # This works only if labels are 0/1

        if len(set(ctcvr_labels)) > 1:
            metrics['ctcvr_auc'] = float(roc_auc_score(ctcvr_labels, ctcvr_preds))
        else:
            metrics['ctcvr_auc'] = 0.5

        return metrics


def simulate_ab_test_data(num_users: int = 1000) -> Tuple[Dict, MetricsTracker]:
    """
    Simulate A/B test data for testing.

    Args:
        num_users: Number of users to simulate

    Returns:
        user_assignments: Dictionary of user group assignments
        tracker: MetricsTracker with simulated data
    """
    random.seed(42)
    np.random.seed(42)

    splitter = ABTestSplitter(seed=42)
    tracker = MetricsTracker()

    user_ids = [f'user_{i:06d}' for i in range(num_users)]
    assignments = splitter.assign_batch(user_ids, treatment_ratio=0.5)

    # Simulate metrics for each user
    for user_id in user_ids:
        group = assignments[user_id]

        # Control group gets baseline metrics
        if group == 'control':
            ctr = np.random.beta(2, 8)  # Average CTR around 20%
            cvr = np.random.beta(1, 9)  # Average CVR around 10%
        # Treatment group gets improved metrics
        else:
            ctr = np.random.beta(3, 7)  # Average CTR around 30%
            cvr = np.random.beta(2, 8)  # Average CVR around 20%

        gmvs = ctr * cvr * np.random.uniform(50, 500)
        session_length = np.random.exponential(300) if random.random() < ctr else np.random.exponential(60)
        items_viewed = int(np.random.poisson(5) * ctr)
        items_purchased = int(np.random.poisson(1) * cvr)
        user_rating = np.random.normal(4.0, 0.5) if items_purchased > 0 else 0

        metrics = {
            'ctr': ctr,
            'cvr': cvr,
            'gmv': gmvs,
            'session_length': session_length,
            'items_viewed': items_viewed,
            'items_purchased': items_purchased,
            'user_rating': user_rating
        }

        tracker.record(group, metrics)

    return assignments, tracker


if __name__ == '__main__':
    print("A/B Testing Framework Test")
    print("=" * 50)

    # Simulate data
    print("\nSimulating A/B test data...")
    assignments, tracker = simulate_ab_test_data(num_users=1000)

    # Evaluate
    print("\nEvaluating results...")
    evaluator = ABTestEvaluator(
        name="Recommendation_Model_Test",
        control_model_name="baseline_dssm",
        treatment_model_name="multimodal_din"
    )

    # Copy tracker data
    evaluator.tracker = tracker

    report = evaluator.evaluate(metrics=['ctr', 'cvr', 'gmv', 'items_purchased'])

    print(f"\nTest: {report['test_name']}")
    print(f"Control Model: {report['control_model']}")
    print(f"Treatment Model: {report['treatment_model']}")
    print(f"\nSample Sizes:")
    print(f"  Control: {report['sample_sizes']['control']}")
    print(f"  Treatment: {report['sample_sizes']['treatment']}")

    print(f"\nResults:")
    for metric, results in report['results'].items():
        print(f"\n  {metric.upper()}:")
        print(f"    Control Mean: {results['control_mean']:.4f}")
        print(f"    Treatment Mean: {results['treatment_mean']:.4f}")
        print(f"    Lift: {results['lift']*100:+.1f}%")
        print(f"    T-test p-value: {results['p_value_ttest']:.4f}")
        print(f"    Significant: {results['significant_ttest']}")

    print(f"\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")