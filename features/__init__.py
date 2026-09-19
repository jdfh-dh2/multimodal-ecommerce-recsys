"""
Features module for Multi-Modal E-Commerce Recommendation System
"""

from .multimodal_encoder import (
    ImageFeatureExtractor,
    TextFeatureExtractor,
    MultiModalFeatureExtractor,
    FeatureProjection
)

__all__ = [
    'ImageFeatureExtractor',
    'TextFeatureExtractor',
    'MultiModalFeatureExtractor',
    'FeatureProjection'
]