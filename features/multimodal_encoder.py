"""
Multi-Modal Feature Extractor
==============================

This module implements feature extraction for product images and text descriptions
using pretrained models (ResNet for images, BERT for text).
"""

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms
import os

# Optional imports with fallbacks
try:
    import torchvision.models as models
    from transformers import BertModel, BertTokenizer
    HAS_TORCHVISION = True
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TORCHVISION = False
    HAS_TRANSFORMERS = False


class ImageFeatureExtractor:
    """
    Extract features from product images using pretrained ResNet50.
    """

    def __init__(self, device='cuda', pretrained=True):
        self.device = device

        if not HAS_TORCHVISION:
            raise ImportError("torchvision is required for image feature extraction")

        # Load pretrained ResNet50
        self.model = models.resnet50(pretrained=pretrained)
        # Remove final classification layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.to(device)
        self.model.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_from_image(self, image_path):
        """
        Extract features from a single image file.

        Args:
            image_path: Path to image file

        Returns:
            features: Feature tensor (2048,)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(image_tensor)

        return features.squeeze().cpu().numpy()

    def extract_from_pil(self, image):
        """
        Extract features from a PIL Image.

        Args:
            image: PIL Image object

        Returns:
            features: Feature tensor (2048,)
        """
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(image_tensor)

        return features.squeeze().cpu().numpy()

    def extract_batch(self, image_paths, batch_size=32):
        """
        Extract features from a batch of images.

        Args:
            image_paths: List of image file paths
            batch_size: Batch size for processing

        Returns:
            features_list: List of feature tensors
        """
        features_list = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []

            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    image_tensor = self.transform(image)
                    batch_tensors.append(image_tensor)
                except Exception as e:
                    print(f"Error loading image {path}: {e}")
                    # Use zero tensor as fallback
                    batch_tensors.append(torch.zeros(3, 224, 224))

            batch = torch.stack(batch_tensors).to(self.device)

            with torch.no_grad():
                features = self.model(batch)

            features_list.extend(features.squeeze(-1).squeeze(-1).cpu().numpy())

        return features_list


class TextFeatureExtractor:
    """
    Extract features from product text descriptions using BERT.
    """

    def __init__(self, device='cuda', model_name='bert-base-uncased'):
        self.device = device

        if not HAS_TRANSFORMERS:
            raise ImportError("transformers is required for text feature extraction")

        # Load pretrained BERT
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def extract_from_text(self, text, max_length=128):
        """
        Extract features from a single text.

        Args:
            text: Input text string
            max_length: Maximum sequence length

        Returns:
            features: Feature tensor (768,)
            embedding: Sentence embedding (768,)
        """
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token embedding as sentence representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        pooled_embedding = outputs.pooler_output

        return cls_embedding.squeeze().cpu().numpy(), pooled_embedding.squeeze().cpu().numpy()

    def extract_batch(self, texts, max_length=128, batch_size=32):
        """
        Extract features from a batch of texts.

        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            batch_size: Batch size for processing

        Returns:
            cls_features_list: List of [CLS] embeddings
            pooled_features_list: List of pooled embeddings
        """
        cls_features_list = []
        pooled_features_list = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize batch
            encoded = self.tokenizer(
                batch_texts,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            cls_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            pooled_features = outputs.pooler_output.cpu().numpy()

            cls_features_list.extend(cls_features)
            pooled_features_list.extend(pooled_features)

        return cls_features_list, pooled_features_list


class MultiModalFeatureExtractor:
    """
    Combined multi-modal feature extractor for e-commerce products.
    """

    def __init__(self, device='cuda'):
        self.device = device

        self.image_extractor = None
        self.text_extractor = None

        if torch.cuda.is_available():
            self.image_extractor = ImageFeatureExtractor(device=device)
            self.text_extractor = TextFeatureExtractor(device=device)

    def extract_image_features(self, image_path):
        """Extract image features."""
        if self.image_extractor is None:
            raise RuntimeError("Image feature extractor not available (torchvision not installed)")
        return self.image_extractor.extract_from_image(image_path)

    def extract_text_features(self, text):
        """Extract text features."""
        if self.text_extractor is None:
            raise RuntimeError("Text feature extractor not available (transformers not installed)")
        return self.text_extractor.extract_from_text(text)

    def extract_multimodal(self, image_path, text):
        """
        Extract combined multi-modal features.

        Args:
            image_path: Path to product image
            text: Product text description

        Returns:
            combined_features: Concatenated feature vector (2048 + 768 = 2816)
        """
        img_features = self.extract_image_features(image_path)
        text_features, _ = self.extract_text_features(text)

        combined = np.concatenate([img_features, text_features])
        return combined

    def extract_multimodal_batch(self, image_paths, texts, batch_size=32):
        """
        Extract multi-modal features for a batch.

        Args:
            image_paths: List of image paths
            texts: List of text descriptions
            batch_size: Batch size

        Returns:
            combined_features: List of combined feature vectors
        """
        img_features = self.image_extractor.extract_batch(image_paths, batch_size)
        text_cls_features, text_pooled_features = self.text_extractor.extract_batch(texts, batch_size=batch_size)

        combined = []
        for img_feat, text_feat in zip(img_features, text_pooled_features):
            combined.append(np.concatenate([img_feat, text_feat]))

        return combined


class FeatureProjection(nn.Module):
    """
    Project multi-modal features to a common embedding space.
    """

    def __init__(self, image_dim=2048, text_dim=768, output_dim=128):
        super(FeatureProjection, self).__init__()

        self.image_projection = nn.Sequential(
            nn.Linear(image_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )

        self.text_projection = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, image_features, text_features):
        """
        Project features to common space.

        Args:
            image_features: Image features (batch, 2048)
            text_features: Text features (batch, 768)

        Returns:
            projected: Projected features (batch, output_dim)
        """
        img_proj = self.image_projection(image_features)
        text_proj = self.text_projection(text_features)

        # Element-wise sum (can also use concatenation)
        return img_proj + text_proj


def create_synthetic_product_data(num_products=1000, output_dir='data/products'):
    """
    Create synthetic product data for testing.

    Args:
        num_products: Number of products to generate
        output_dir: Output directory for data

    Returns:
        product_data: List of product dictionaries
    """
    os.makedirs(output_dir, exist_ok=True)

    product_data = []

    categories = ['Electronics', 'Clothing', 'Home', 'Books', 'Sports']
    brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE']

    for i in range(num_products):
        product = {
            'id': f'product_{i:06d}',
            'name': f'Product {i} - {categories[i % len(categories)]}',
            'description': f'This is a high-quality {categories[i % len(categories)]} product from {brands[i % len(brands)]}. Perfect for everyday use.',
            'category': categories[i % len(categories)],
            'brand': brands[i % len(brands)],
            'price': round(np.random.uniform(9.99, 999.99), 2),
            'image_path': f'{output_dir}/image_{i:06d}.jpg'
        }
        product_data.append(product)

    return product_data


if __name__ == '__main__':
    # Test the extractors
    print("Multi-Modal Feature Extractor Test")
    print("=" * 50)

    # Check availability
    print(f"torchvision available: {HAS_TORCHVISION}")
    print(f"transformers available: {HAS_TRANSFORMERS}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if HAS_TORCHVISION and HAS_TRANSFORMERS and torch.cuda.is_available():
        # Test image extractor
        print("\nTesting Image Feature Extractor...")
        img_extractor = ImageFeatureExtractor()

        # Create a random test image
        test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_feat = img_extractor.extract_from_pil(test_image)
        print(f"Image feature shape: {img_feat.shape}")

        # Test text extractor
        print("\nTesting Text Feature Extractor...")
        text_extractor = TextFeatureExtractor()
        cls_feat, pooled_feat = text_extractor.extract_from_text("This is a great product!")
        print(f"Text CLS feature shape: {cls_feat.shape}")
        print(f"Text pooled feature shape: {pooled_feat.shape}")

        # Test combined extractor
        print("\nTesting Multi-Modal Feature Extractor...")
        multimodal = MultiModalFeatureExtractor()
        print("Multi-modal extractor initialized successfully")
    else:
        print("\nSkipping tests - required libraries not available")