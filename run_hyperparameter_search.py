#!/usr/bin/env python3
"""
Arabic Writer Identification - Hyperparameter Search (Auto Multi-GPU)
Bayesian Optimization using Optuna with Automatic GPU Detection

Task: Find optimal hyperparameters for both ConvNeXt and Transformer models
Data: Historical Arabic manuscript line images
Optimization: Bayesian search with early stopping

GPU Modes (Automatic):
- 1 GPU: Sequential trials on single GPU
- 2+ GPUs: Parallel trials (one trial per GPU) for 4x speedup

Usage:
    python run_hyperparameter_search.py \\
        --data_root /path/to/data \\
        --checkpoint_dir /path/to/checkpoints \\
        --n_trials 24 \\
        --use_all_writers

Author: Generated for HPC server
Date: 2025-12-27
"""

# ============================================================
# SECTION 1: IMPORTS & SETUP
# ============================================================

import os
import sys
import random
import pickle
import json
import argparse
import subprocess
import signal
import time
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.models as models
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler

# ML utilities
from sklearn.metrics import f1_score, average_precision_score

# Optuna for Bayesian optimization
import optuna
from optuna.trial import TrialState
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# SEED SETTING
# ============================================================

def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ============================================================
# BASE CONFIGURATION
# ============================================================

class BaseConfig:
    """Base configuration with fixed parameters."""
    
    # Data paths - UPDATE THESE!
    DATA_ROOT = '/path/to/extracted_lines'  # e.g., /project/mamro/extracted_lines
    CHECKPOINT_DIR = '/path/to/checkpoints/hyperparam_search'
    
    # Writers will be auto-discovered from DATA_ROOT/train directory
    # This prevents spelling mistakes
    ALL_WRITERS = None  # Will be set during initialization
    
    # Option: Use subset of writers for faster iteration
    USE_ALL_WRITERS = True  # Set to False to use first 7 writers
    NUM_WRITERS_SUBSET = 7  # If USE_ALL_WRITERS=False, use this many writers
    
    # Image preprocessing (fixed)
    IMG_HEIGHT = 160
    IMG_WIDTH = 1760
    MIN_INK_RATIO = 0
    MIN_RESIZED_HEIGHT = 0
    
    # Augmentation (fixed)
    AUG_BINARIZE_PROB = 0.5
    AUG_ROTATION_DEGREES = 2
    AUG_ELASTIC_ALPHA = 0
    AUG_GAUSSIAN_NOISE_STD = 0.02
    AUG_GAUSSIAN_BLUR_PROB = 0.2
    AUG_BRIGHTNESS_RANGE = 0.1
    AUG_CONTRAST_RANGE = 0.1
    
    # Fixed training parameters
    EPOCHS = 70  # Per trial (increased for better convergence)
    EARLY_STOPPING_PATIENCE = 15
    EARLY_STOPPING_MIN_DELTA = 0
    MAX_PAGES_PER_WRITER_VAL = 20
    
    # Training
    NUM_WORKERS = 4
    
    # Mixed precision
    USE_AMP = True
    
    # ImageNet normalization
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # Vision Transformer specific (if using ViT)
    VIT_PATCH_HEIGHT = 16
    VIT_PATCH_WIDTH = 32
    VIT_HIDDEN_DIM = 768
    VIT_NUM_HEADS = 12
    VIT_NUM_LAYERS = 12
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Quick test mode (set via command line --quick_test)
    QUICK_TEST = False


# ============================================================
# HYPERPARAMETER SEARCH SPACE
# ============================================================

class HyperparameterSpace:
    """Define the hyperparameter search space optimized for 12 trials."""
    
    @staticmethod
    def suggest_hyperparameters(trial, quick_test=False):
        """
        Suggest hyperparameters for a trial.

        Strategy for 12 trials:
        - Focus on most impactful parameters
        - Strong regularization to combat overfitting
        - Model-specific configurations for ConvNeXt vs ViT

        Args:
            trial: Optuna trial object
            quick_test: If True, use smaller batch size for fast testing
        """
        
        # ========== MODEL ARCHITECTURE ==========
        model_type = trial.suggest_categorical('model_type', ['convnext', 'vit'])
        
        # ========== CORE LEARNING PARAMETERS ==========
        # Learning rate (log scale) - critical parameter
        learning_rate = trial.suggest_float('learning_rate', 5e-5, 5e-4, log=True)
        
        # Weight decay (strong regularization against overfitting)
        weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-3, log=True)
        
        # ========== DROPOUT (Strong regularization) ==========
        # Higher dropout to combat overfitting
        dropout_embedding = trial.suggest_float('dropout_embedding', 0.3, 0.5)
        dropout_classifier = trial.suggest_float('dropout_classifier', 0.5, 0.7)
        
        # ViT-specific: attention dropout
        if model_type == 'vit':
            dropout_attention = trial.suggest_float('dropout_attention', 0.1, 0.3)
        else:
            dropout_attention = 0.0
        
        # ========== LABEL SMOOTHING (Regularization) ==========
        label_smoothing = trial.suggest_float('label_smoothing', 0.05, 0.15)
        
        # ========== LOSS CONFIGURATION ==========
        # Strategy: Use 3 losses as in ConvNeXt notebook
        # - Focal Loss OR CrossEntropy (not both)
        # - Triplet Loss (optional)
        # - MixUp with CrossEntropy only (not with Focal/Triplet)
        
        use_focal = trial.suggest_categorical('use_focal_loss', [True, False])
        
        if use_focal:
            # Focal Loss parameters
            focal_gamma = trial.suggest_float('focal_gamma', 1.5, 2.5)
            focal_alpha = trial.suggest_float('focal_alpha', 0.75, 1.25)
            # No MixUp with Focal
            mixup_alpha = 0.0
        else:
            # Use CrossEntropy with MixUp
            focal_gamma = 2.0  # dummy
            focal_alpha = 1.0  # dummy
            mixup_alpha = trial.suggest_float('mixup_alpha', 0.1, 0.4)
        
        # Triplet loss (metric learning)
        use_triplet = trial.suggest_categorical('use_triplet_loss', [True, False])
        if use_triplet:
            triplet_weight = trial.suggest_float('triplet_weight', 0.05, 0.2)
            triplet_margin = trial.suggest_float('triplet_margin', 0.4, 0.6)
        else:
            triplet_weight = 0.0
            triplet_margin = 0.5
        
        # ========== MODEL-SPECIFIC PARAMETERS ==========
        # Use unified search space for all models (Optuna requirement)
        # Suggest from full range, then apply model-specific constraints
        embedding_dim_suggested = trial.suggest_categorical('embedding_dim', [768, 1024])
        freeze_backbone = trial.suggest_categorical('freeze_backbone', [True, False])

        if freeze_backbone:
            # Unified range for all models (Optuna requirement)
            freeze_layers = trial.suggest_int('freeze_layers', 6, 25)
        else:
            freeze_layers = 0

        # Apply model-specific constraints
        if model_type == 'vit':
            # ViT only supports 768 embedding dimension
            embedding_dim = 768
        else:
            # ConvNeXt can use suggested dimension
            embedding_dim = embedding_dim_suggested
        
        # ========== TRAINING SCHEDULE ==========
        # Warmup epochs (helps with stability)
        warmup_epochs = trial.suggest_int('warmup_epochs', 5, 10)
        
        # Gradient clipping (prevent exploding gradients)
        gradient_clip_norm = trial.suggest_float('gradient_clip_norm', 0.8, 1.5)
        
        # ========== BATCH SIZE ==========
        # Fixed batch size for 12 trials (reduce search space)
        # Can adjust based on GPU memory
        if quick_test:
            batch_size = 32  # Small batch for quick testing
        else:
            batch_size = 128  # Full batch for production
        
        return {
            'model_type': model_type,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'batch_size': batch_size,
            'dropout_embedding': dropout_embedding,
            'dropout_classifier': dropout_classifier,
            'dropout_attention': dropout_attention,  # ViT only
            'label_smoothing': label_smoothing,
            'use_focal_loss': use_focal,
            'focal_gamma': focal_gamma,
            'focal_alpha': focal_alpha,
            'mixup_alpha': mixup_alpha,
            'use_triplet_loss': use_triplet,
            'triplet_weight': triplet_weight,
            'triplet_margin': triplet_margin,
            'embedding_dim': embedding_dim,
            'freeze_backbone': freeze_backbone,
            'freeze_layers': freeze_layers,
            'warmup_epochs': warmup_epochs,
            'gradient_clip_norm': gradient_clip_norm,
        }
    
    @staticmethod
    def print_search_space_summary():
        """Print summary of search space."""
        print("\n" + "="*70)
        print("HYPERPARAMETER SEARCH SPACE SUMMARY")
        print("="*70)
        print("\n🎯 OPTIMIZATION STRATEGY:")
        print("  • 12 trials with Bayesian optimization (TPE sampler)")
        print("  • Strong regularization to combat overfitting")
        print("  • Model-specific configurations (ConvNeXt vs ViT)")
        print("\n📊 CORE PARAMETERS (12 trials):")
        print("  1. Model Type: ConvNeXt vs ViT")
        print("  2. Learning Rate: [5e-5, 5e-4]")
        print("  3. Weight Decay: [1e-4, 1e-3]")
        print("  4. Dropout Embedding: [0.3, 0.5]")
        print("  5. Dropout Classifier: [0.5, 0.7]")
        print("  6. Label Smoothing: [0.05, 0.15]")
        print("\n🔥 LOSS STRATEGY:")
        print("  • Option A: Focal Loss + Triplet")
        print("  • Option B: CrossEntropy + MixUp + Triplet")
        print("  • Focal gamma: [1.5, 2.5], alpha: [0.75, 1.25]")
        print("  • MixUp alpha: [0.1, 0.4] (only with CE)")
        print("  • Triplet weight: [0.05, 0.2], margin: [0.4, 0.6]")
        print("\n⚙️  MODEL-SPECIFIC:")
        print("  ConvNeXt:")
        print("    - Embedding: [768, 1024]")
        print("    - Freeze layers: [15, 25] (if enabled)")
        print("  ViT:")
        print("    - Embedding: 768 (fixed)")
        print("    - Freeze layers: [6, 10] (if enabled)")
        print("    - Attention dropout: [0.1, 0.3]")
        print("\n📦 SAMPLER (Balanced):")
        print("  • Quota per writer: 300 (fixed)")
        print("  • Max exposure per line: 2 (fixed)")
        print("\n🎓 TRAINING:")
        print("  • Batch size: 128 (fixed)")
        print("  • Warmup: [5, 10] epochs")
        print("  • Gradient clip: [0.8, 1.5]")
        print("  • Max epochs: 30 per trial")
        print("  • Early stopping: 15 epochs patience")
        print("="*70 + "\n")


# ============================================================
# CONFIGURATION MERGER
# ============================================================

class TrialConfig(BaseConfig):
    """Configuration for a single trial."""
    
    def __init__(self, hyperparams):
        super().__init__()

        # Merge hyperparameters
        for key, value in hyperparams.items():
            setattr(self, key.upper(), value)

        # Writers and NUM_CLASSES will be set after data indexing
        self.WRITERS = None
        self.NUM_CLASSES = None


print("✅ Section 1: Imports and setup complete")


# ============================================================
# SECTION 2: DATA INDEXING & PREPROCESSING
# ============================================================

def discover_writers(data_root, split='train'):
    """
    Auto-discover writer names from the data directory.
    Prevents spelling errors by reading directly from filesystem.
    
    Args:
        data_root: Path to Mirath_extracted_lines
        split: Which split to use for discovery (default: train)
    
    Returns:
        List of writer names (sorted)
    """
    split_path = Path(data_root) / split
    
    if not split_path.exists():
        raise FileNotFoundError(f"Split path not found: {split_path}")
    
    writers = sorted([d.name for d in split_path.iterdir() if d.is_dir()])
    
    print(f"\n📚 Auto-discovered {len(writers)} writers from {split}:")
    for i, writer in enumerate(writers, 1):
        print(f"  {i:2d}. {writer}")
    
    return writers


def build_index(data_root, splits=['train', 'val'], selected_writers=None):
    """
    Build in-memory index of all line images.
    
    Data structure:
    - Mirath_extracted_lines/
      ├── train/
      │   ├── writer_1/
      │   │   ├── image1/          # Folder with extracted lines
      │   │   │   ├── line_001.png
      │   │   │   ├── line_002.png
      │   │   │   └── ...
      │   │   └── book_1/
      │   │       └── page1/       # Folder with extracted lines
      │   │           ├── line_001.png
      │   │           └── ...
      │   └── writer_2/...
      ├── val/...
      └── test/...
    
    Args:
        data_root: Path to Mirath_extracted_lines
        splits: List of splits to index
        selected_writers: Optional list of writer names to use (subset)
    
    Returns:
        writers: List of writer names
        writer2idx: Dict mapping writer name to index
        split_data: Dict containing lines and pages for each split
    """
    data_root = Path(data_root)
    
    # Discover all writers from train split
    all_writers = discover_writers(data_root, split='train')
    
    # Use selected writers or all writers
    if selected_writers is not None:
        writers = [w for w in selected_writers if w in all_writers]
        if len(writers) != len(selected_writers):
            missing = set(selected_writers) - set(writers)
            print(f"\n⚠️  Warning: Some selected writers not found: {missing}")
    else:
        writers = all_writers
    
    writer2idx = {w: i for i, w in enumerate(writers)}
    print(f"\n✅ Using {len(writers)} writers for training")
    
    split_data = {}
    
    for split in splits:
        print(f"\n{'='*70}")
        print(f"INDEXING: {split.upper()}")
        print(f"{'='*70}")
        
        split_path = data_root / split
        if not split_path.exists():
            print(f"⚠️  {split_path} does not exist, skipping...")
            continue
        
        lines = []
        pages = defaultdict(list)
        writer_stats = defaultdict(lambda: {'pages': 0, 'lines': 0, 'empty_pages': 0})
        
        total_empty_folders = 0
        
        for writer in tqdm(writers, desc=f"Indexing {split}"):
            writer_path = split_path / writer
            if not writer_path.exists():
                print(f"  ⚠️  Writer not found in {split}: {writer}")
                continue
            
            writer_idx = writer2idx[writer]
            
            # Find all leaf directories (page folders with line images)
            # Each leaf directory is a "page" (either direct image folder or book/page folder)
            for page_folder in writer_path.rglob('*'):
                if not page_folder.is_dir():
                    continue
                
                # Check if this is a leaf directory (no subdirectories)
                subdirs = [d for d in page_folder.iterdir() if d.is_dir()]
                if len(subdirs) > 0:
                    # This is a container (book folder), not a page
                    continue
                
                # This is a leaf directory - check for line images
                line_images = list(page_folder.glob('*.png'))
                line_images.extend(list(page_folder.glob('*.jpg')))
                line_images.extend(list(page_folder.glob('*.jpeg')))
                
                if len(line_images) == 0:
                    # Empty page folder
                    writer_stats[writer]['empty_pages'] += 1
                    total_empty_folders += 1
                    continue
                
                # Valid page with images
                # Create unique page ID
                page_id = f"{writer}_{page_folder.relative_to(writer_path)}".replace('/', '_').replace('\\', '_')
                writer_stats[writer]['pages'] += 1
                
                # Add lines from this page
                page_line_indices = []
                for line_img in sorted(line_images):
                    line_idx = len(lines)
                    lines.append({
                        'path': str(line_img),
                        'writer_idx': writer_idx,
                        'page_id': page_id
                    })
                    page_line_indices.append(line_idx)
                    writer_stats[writer]['lines'] += 1
                
                pages[page_id] = page_line_indices
        
        split_data[split] = {
            'lines': lines,
            'pages': dict(pages),
            'stats': dict(writer_stats)
        }
        
        # Print statistics
        print(f"\n{split.upper()} Statistics:")
        print(f"  Total lines: {len(lines):,}")
        print(f"  Total pages (with lines): {len(pages):,}")
        print(f"  Empty folders skipped: {total_empty_folders}")
        print(f"\n  Per-writer:")
        print(f"  {'Writer':<25} {'Pages':<8} {'Lines':<8} {'Empty':<8} {'Avg Lines/Page':<15}")
        print(f"  {'-'*70}")
        
        for writer in writers:
            stats = writer_stats[writer]
            avg_lines = stats['lines'] / stats['pages'] if stats['pages'] > 0 else 0
            print(f"  {writer:<25} {stats['pages']:<8} {stats['lines']:<8} "
                  f"{stats['empty_pages']:<8} {avg_lines:<15.1f}")
    
    return writers, writer2idx, split_data


# ============================================================
# PREPROCESSING FUNCTIONS
# ============================================================

def preprocess_line_image(img_path, config, apply_binarize=False, is_train=False):
    """
    Arabic-aware preprocessing with augmentation.
    All parameters come from config object.
    
    Augmentation strategy (training only):
    - Rotation (±config.AUG_ROTATION_DEGREES)
    - Gaussian blur (config.AUG_GAUSSIAN_BLUR_PROB)
    - Brightness/contrast (config.AUG_BRIGHTNESS_RANGE, config.AUG_CONTRAST_RANGE)
    - Gaussian noise (config.AUG_GAUSSIAN_NOISE_STD)
    - Optional binarization (config.AUG_BINARIZE_PROB)
    
    Args:
        img_path: Path to line image
        config: Configuration object with all parameters
        apply_binarize: Whether to apply binarization
        is_train: Whether this is training (enables augmentation)
    
    Returns:
        Tensor of shape (3, IMG_HEIGHT, IMG_WIDTH) or None if failed
    """
    try:
        # Read as RGB
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w = img.shape[:2]
        
        # Rotate if portrait (make horizontal)
        if h > w * 1.5:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            h, w = img.shape[:2]
        
        # Resize maintaining aspect ratio
        scale = min(config.IMG_HEIGHT / h, config.IMG_WIDTH / w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # Filter: check resized height (from config)
        if new_h < config.MIN_RESIZED_HEIGHT:
            return None
        
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # ========== TRAINING AUGMENTATIONS ==========
        if is_train:
            # 1. Rotation (from config)
            if random.random() < 0.3:
                angle = random.uniform(-config.AUG_ROTATION_DEGREES, config.AUG_ROTATION_DEGREES)
                center = (new_w // 2, new_h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img_resized = cv2.warpAffine(img_resized, M, (new_w, new_h),
                                            borderMode=cv2.BORDER_REPLICATE)
            
            # 2. Gaussian blur (from config)
            if random.random() < config.AUG_GAUSSIAN_BLUR_PROB:
                kernel_size = random.choice([3, 5])
                img_resized = cv2.GaussianBlur(img_resized, (kernel_size, kernel_size), 0)
            
            # 3. Brightness/Contrast adjustment (from config)
            if random.random() < 0.3:
                alpha = 1.0 + random.uniform(-config.AUG_CONTRAST_RANGE, config.AUG_CONTRAST_RANGE)
                beta = random.uniform(-config.AUG_BRIGHTNESS_RANGE * 255, config.AUG_BRIGHTNESS_RANGE * 255)
                img_resized = cv2.convertScaleAbs(img_resized, alpha=alpha, beta=beta)
            
            # 4. Gaussian noise (from config)
            if random.random() < 0.2:
                noise = np.random.normal(0, config.AUG_GAUSSIAN_NOISE_STD * 255, img_resized.shape)
                img_resized = np.clip(img_resized.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # ========== BINARIZATION (Optional) ==========
        if apply_binarize:
            gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img_resized = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        
        # ========== PAD TO TARGET SIZE ==========
        # White background padding
        padded = np.ones((config.IMG_HEIGHT, config.IMG_WIDTH, 3), dtype=np.uint8) * 255
        padded[:new_h, :new_w] = img_resized
        
        # ========== CONVERT TO TENSOR ==========
        tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
        
        # Normalize with ImageNet statistics (from config)
        normalize = transforms.Normalize(
            mean=config.IMAGENET_MEAN,
            std=config.IMAGENET_STD
        )
        tensor = normalize(tensor)
        
        return tensor
        
    except Exception as e:
        return None


# ============================================================
# DATASET CLASS
# ============================================================

class ArabicLineDataset(Dataset):
    """
    Dataset for Arabic line images with augmentation.
    
    Args:
        lines: List of line dictionaries
        config: Configuration object
        is_train: Whether this is training set (enables augmentation)
    """
    
    def __init__(self, lines, config, is_train=False):
        self.lines = lines
        self.config = config
        self.is_train = is_train
        self.failed_count = 0
    
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        line_info = self.lines[idx]
        
        # Decide whether to apply binarization (from config)
        apply_binarize = self.is_train and (random.random() < self.config.AUG_BINARIZE_PROB)
        
        # Preprocess image (all parameters from config)
        img_tensor = preprocess_line_image(
            line_info['path'],
            config=self.config,
            apply_binarize=apply_binarize,
            is_train=self.is_train
        )
        
        # Fallback to white image if preprocessing failed
        if img_tensor is None:
            self.failed_count += 1
            img_tensor = torch.ones(3, self.config.IMG_HEIGHT, self.config.IMG_WIDTH)
            normalize = transforms.Normalize(
                mean=self.config.IMAGENET_MEAN,
                std=self.config.IMAGENET_STD
            )
            img_tensor = normalize(img_tensor)
        
        return {
            'image': img_tensor,
            'writer_idx': line_info['writer_idx'],
            'page_id': line_info['page_id'],
            'path': line_info['path']
        }


print("✅ Section 2: Data indexing and preprocessing complete")


# ============================================================
# SECTION 3: WRITER-BALANCED SAMPLER
# ============================================================

class WriterBalancedSampler(Sampler):
    """
    Simplified writer-balanced sampler with exposure tracking.

    Algorithm:
    1. Each writer gets 300 lines per epoch (QUOTA_PER_WRITER)
    2. Sample lines with lowest exposure first (fairness)
    3. Maximum exposure per line: 2 (MAX_EXPOSURE)
    4. Track coverage: which lines have been seen

    Args:
        lines: List of line dictionaries
        quota_per_writer: Lines per writer per epoch (default: 300)
        max_exposure: Maximum times a line can be sampled (default: 2)
    """

    def __init__(self, lines, quota_per_writer=300, max_exposure=2):
        self.lines = lines
        self.quota_per_writer = quota_per_writer
        self.max_exposure = max_exposure

        # Group lines by writer
        self.writer_lines = defaultdict(list)
        for idx, line in enumerate(lines):
            writer_idx = line['writer_idx']
            self.writer_lines[writer_idx].append(idx)

        self.writers = sorted(self.writer_lines.keys())
        self.num_writers = len(self.writers)

        # Initialize exposure tracking (0 = never seen)
        self.line_exposure = {idx: 0 for idx in range(len(lines))}

        # Statistics
        self.total_samples_per_epoch = self.quota_per_writer * self.num_writers
        self.current_epoch = 0

    def __iter__(self):
        """
        Generate indices for one epoch.

        Strategy:
        1. For each writer, get their lines sorted by exposure (lowest first)
        2. Sample up to quota_per_writer lines
        3. Skip lines that have reached max_exposure
        4. If quota not met and all lines used, repeat from lowest exposure
        5. Update exposure counts
        """
        epoch_indices = []

        self.current_epoch += 1

        # For each writer, sample up to quota
        for writer_idx in self.writers:
            writer_line_indices = self.writer_lines[writer_idx]
            total_available = len(writer_line_indices)

            # Sort lines by exposure (lowest first)
            lines_sorted = sorted(writer_line_indices, key=lambda idx: self.line_exposure[idx])

            writer_epoch_indices = []

            # Sample up to quota
            for line_idx in lines_sorted:
                if len(writer_epoch_indices) >= self.quota_per_writer:
                    break

                # Only sample if below max exposure
                if self.line_exposure[line_idx] < self.max_exposure:
                    writer_epoch_indices.append(line_idx)
                    self.line_exposure[line_idx] += 1

            # If quota not met, check if we can repeat any lines
            if len(writer_epoch_indices) < self.quota_per_writer:
                # Try to fill quota by repeating lines (if allowed)
                for line_idx in lines_sorted:
                    if len(writer_epoch_indices) >= self.quota_per_writer:
                        break

                    if self.line_exposure[line_idx] < self.max_exposure:
                        writer_epoch_indices.append(line_idx)
                        self.line_exposure[line_idx] += 1

            epoch_indices.extend(writer_epoch_indices)

        # Shuffle final indices for random batching
        random.shuffle(epoch_indices)

        return iter(epoch_indices)

    def __len__(self):
        """Total samples per epoch."""
        return self.total_samples_per_epoch

    def get_coverage_stats(self, writers):
        """
        Calculate coverage statistics: how many lines have been trained on.

        Returns:
            Dictionary with overall and per-writer coverage statistics
        """
        overall_stats = {
            'total_lines': len(self.lines),
            'seen_lines': 0,
            'unseen_lines': 0,
            'coverage_pct': 0.0,
            'avg_exposure': 0.0,
            'max_exposure_reached': 0,
        }

        per_writer_stats = {}

        # Calculate overall stats
        seen_count = sum(1 for exp in self.line_exposure.values() if exp > 0)
        max_exp_count = sum(1 for exp in self.line_exposure.values() if exp >= self.max_exposure)

        overall_stats['seen_lines'] = seen_count
        overall_stats['unseen_lines'] = len(self.lines) - seen_count
        overall_stats['coverage_pct'] = (seen_count / len(self.lines) * 100) if len(self.lines) > 0 else 0.0
        overall_stats['avg_exposure'] = np.mean(list(self.line_exposure.values())) if self.line_exposure else 0.0
        overall_stats['max_exposure_reached'] = max_exp_count

        # Per-writer stats
        for writer_idx in self.writers:
            writer_line_indices = self.writer_lines[writer_idx]
            total_lines = len(writer_line_indices)
            seen_lines = sum(1 for idx in writer_line_indices if self.line_exposure[idx] > 0)
            unseen_lines = total_lines - seen_lines
            coverage_pct = (seen_lines / total_lines * 100) if total_lines > 0 else 0.0
            avg_exposure = np.mean([self.line_exposure[idx] for idx in writer_line_indices]) if total_lines > 0 else 0.0
            max_exp_reached = sum(1 for idx in writer_line_indices if self.line_exposure[idx] >= self.max_exposure)

            per_writer_stats[writer_idx] = {
                'writer_name': writers[writer_idx],
                'total_lines': total_lines,
                'seen_lines': seen_lines,
                'unseen_lines': unseen_lines,
                'coverage_pct': coverage_pct,
                'avg_exposure': avg_exposure,
                'max_exposure_reached': max_exp_reached,
                'all_lines_seen': (unseen_lines == 0)
            }

        return overall_stats, per_writer_stats

    def print_coverage_report(self, writers):
        """Print detailed coverage report."""
        overall_stats, per_writer_stats = self.get_coverage_stats(writers)

        print(f"\n{'='*80}")
        print("TRAINING COVERAGE REPORT")
        print(f"{'='*80}")
        print(f"Total Epochs: {self.current_epoch}")
        print(f"Quota per Writer per Epoch: {self.quota_per_writer}")
        print(f"Max Exposure Allowed: {self.max_exposure}")
        print(f"\n{'─'*80}")
        print("OVERALL STATISTICS")
        print(f"{'─'*80}")
        print(f"Total Lines in Dataset: {overall_stats['total_lines']:,}")
        print(f"Lines Trained On (≥1 time): {overall_stats['seen_lines']:,}")
        print(f"Lines Never Seen: {overall_stats['unseen_lines']:,}")
        print(f"Coverage: {overall_stats['coverage_pct']:.2f}%")
        print(f"Average Exposure per Line: {overall_stats['avg_exposure']:.2f}")
        print(f"Lines at Max Exposure ({self.max_exposure}x): {overall_stats['max_exposure_reached']:,}")

        # Check if all lines trained
        all_trained = overall_stats['unseen_lines'] == 0
        if all_trained:
            print(f"\n✅ ALL LINES HAVE BEEN TRAINED ON!")
        else:
            print(f"\n⚠️  NOT ALL LINES TRAINED - {overall_stats['unseen_lines']:,} lines never seen")

        print(f"\n{'─'*80}")
        print("PER-WRITER STATISTICS")
        print(f"{'─'*80}")
        print(f"{'Writer':<20} {'Total':<8} {'Seen':<8} {'Unseen':<8} {'Coverage':<10} {'Avg Exp':<8} {'All Seen?'}")
        print(f"{'-'*80}")

        for writer_idx in sorted(per_writer_stats.keys()):
            stats = per_writer_stats[writer_idx]
            writer_name = stats['writer_name'][:18]  # Truncate if too long
            all_seen_marker = "✅" if stats['all_lines_seen'] else "❌"

            print(f"{writer_name:<20} "
                  f"{stats['total_lines']:<8} "
                  f"{stats['seen_lines']:<8} "
                  f"{stats['unseen_lines']:<8} "
                  f"{stats['coverage_pct']:>7.2f}% "
                  f"{stats['avg_exposure']:>8.2f} "
                  f"{all_seen_marker}")

        print(f"{'='*80}\n")

    def print_sampler_info(self, writers):
        """Print sampler configuration and statistics."""
        print(f"\n{'='*70}")
        print("WRITER-BALANCED SAMPLER CONFIGURATION")
        print(f"{'='*70}")
        print(f"  Quota per Writer per Epoch: {self.quota_per_writer}")
        print(f"  Maximum Exposure per Line:  {self.max_exposure}")
        print(f"  Number of Writers:          {self.num_writers}")
        print(f"  Total Samples per Epoch:    {self.total_samples_per_epoch:,}")
        print(f"\n  Per-writer line counts:")
        print(f"  {'Writer':<25} {'Total Lines':<12} {'Lines per Epoch'}")
        print(f"  {'-'*70}")

        for writer_idx in self.writers:
            writer_name = writers[writer_idx]
            total_lines = len(self.writer_lines[writer_idx])
            print(f"  {writer_name:<25} {total_lines:<12,} {self.quota_per_writer}")

        print(f"{'='*70}\n")


print("✅ Section 3: Writer-balanced sampler complete")


# ============================================================
# SECTION 4: MODEL ARCHITECTURES
# ============================================================

# ============================================================
# CONVNEXT-BASE MODEL
# ============================================================

class ConvNeXtWriterID(nn.Module):
    """
    ConvNeXt-Base for writer identification.
    
    Architecture:
    - Backbone: ConvNeXt-Base (pretrained on ImageNet)
    - Embedding head: Feature extraction with regularization
    - Classifier head: Multi-layer classifier with dropout
    
    Features:
    - Configurable backbone freezing (freeze_layers parameter)
    - Strong dropout for regularization
    - Separate embedding and classification paths
    
    Args:
        num_classes: Number of writer classes
        embedding_dim: Dimension of embedding space
        dropout_embedding: Dropout rate for embedding head
        dropout_classifier: Dropout rate for classifier head
        freeze_backbone: Whether to freeze backbone layers
        freeze_layers: Number of layers to freeze (from start)
        pretrained_path: Optional path to pretrained weights
    """
    
    def __init__(self, num_classes, embedding_dim=1024,
                 dropout_embedding=0.4, dropout_classifier=0.6,
                 freeze_backbone=False, freeze_layers=20,
                 pretrained_path=None):
        super().__init__()
        
        # Load ConvNeXt-Base
        try:
            if pretrained_path and Path(pretrained_path).exists():
                print(f"  Loading pretrained weights from {pretrained_path}")
                convnext = models.convnext_base(pretrained=False)
                state_dict = torch.load(pretrained_path, map_location='cpu')
                convnext.load_state_dict(state_dict)
            else:
                print("  Loading ConvNeXt-Base with ImageNet weights...")
                convnext = models.convnext_base(pretrained=True)
        except Exception as e:
            print(f"  ⚠️  Could not load pretrained weights: {e}")
            print("  Training from scratch")
            convnext = models.convnext_base(pretrained=False)
        
        # Remove classifier, keep feature extractor
        self.backbone = nn.Sequential(*list(convnext.children())[:-1])
        
        # Freeze backbone layers if requested
        if freeze_backbone and freeze_layers > 0:
            params_list = list(self.backbone.parameters())
            for param in params_list[:freeze_layers]:
                param.requires_grad = False
            print(f"  Frozen first {freeze_layers} backbone layers")
        
        # ConvNeXt-Base outputs 1024 features
        # Embedding head: Deep projection with regularization
        self.embedding = nn.Sequential(
            nn.Dropout(p=dropout_embedding),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_embedding),
            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        # Classifier head: Multi-layer with strong regularization
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_classifier),
            nn.Linear(embedding_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_classifier),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x, return_embedding=False):
        """
        Forward pass.
        
        Args:
            x: Input images (B, 3, H, W)
            return_embedding: If True, return normalized embeddings for retrieval
        
        Returns:
            If return_embedding=False: (logits, embeddings)
            If return_embedding=True: normalized_embeddings
        """
        # Backbone: (B, 3, H, W) → (B, 1024, 1, 1)
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # (B, 1024)
        
        # Embedding: (B, 1024) → (B, embedding_dim)
        embedding = self.embedding(features)
        
        if return_embedding:
            # L2-normalize for retrieval/metric learning
            return F.normalize(embedding, p=2, dim=1)
        
        # Classification: (B, embedding_dim) → (B, num_classes)
        logits = self.classifier(embedding)
        
        return logits, embedding


# ============================================================
# VISION TRANSFORMER MODEL
# ============================================================

class VisionTransformerWriterID(nn.Module):
    """
    Vision Transformer for writer identification with rectangular patches.
    
    Architecture:
    - Patch embedding with rectangular patches (16×32)
    - Transformer encoder (12 layers, 12 heads)
    - Embedding head with regularization
    - Classifier head with dropout
    
    Features:
    - Rectangular patches optimized for line images (160×1760)
    - Configurable layer freezing
    - Attention dropout for regularization
    - Learnable positional embeddings
    
    Args:
        num_classes: Number of writer classes
        img_height: Image height (default: 160)
        img_width: Image width (default: 1760)
        patch_height: Patch height (default: 16)
        patch_width: Patch width (default: 32)
        hidden_dim: Transformer hidden dimension (default: 768)
        num_heads: Number of attention heads (default: 12)
        num_layers: Number of transformer layers (default: 12)
        embedding_dim: Output embedding dimension (default: 768)
        dropout_embedding: Dropout for embedding head
        dropout_classifier: Dropout for classifier head
        dropout_attention: Dropout in attention layers
        freeze_backbone: Whether to freeze transformer layers
        freeze_layers: Number of transformer layers to freeze
    """
    
    def __init__(self, num_classes,
                 img_height=160, img_width=1760,
                 patch_height=16, patch_width=32,
                 hidden_dim=768, num_heads=12, num_layers=12,
                 embedding_dim=768,
                 dropout_embedding=0.3, dropout_classifier=0.5,
                 dropout_attention=0.1,
                 freeze_backbone=False, freeze_layers=6):
        super().__init__()
        
        self.img_height = img_height
        self.img_width = img_width
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.hidden_dim = hidden_dim
        
        # Calculate number of patches
        self.num_patches_h = img_height // patch_height  # 160/16 = 10
        self.num_patches_w = img_width // patch_width     # 1760/32 = 55
        self.num_patches = self.num_patches_h * self.num_patches_w  # 550
        
        print(f"  ViT Configuration:")
        print(f"    Image size: {img_height}×{img_width}")
        print(f"    Patch size: {patch_height}×{patch_width}")
        print(f"    Number of patches: {self.num_patches_h}×{self.num_patches_w} = {self.num_patches}")
        
        # Patch embedding: Rectangular patches
        patch_dim = 3 * patch_height * patch_width
        self.patch_embedding = nn.Linear(patch_dim, hidden_dim)
        
        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, hidden_dim))
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_attention,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Freeze transformer layers if requested
        if freeze_backbone and freeze_layers > 0:
            for i, layer in enumerate(self.transformer.layers):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
            print(f"  Frozen first {freeze_layers} transformer layers")
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Embedding head
        self.embedding = nn.Sequential(
            nn.Dropout(p=dropout_embedding),
            nn.Linear(hidden_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_embedding)
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_classifier),
            nn.Linear(embedding_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_classifier),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x, return_embedding=False):
        """
        Forward pass.
        
        Args:
            x: Input images (B, 3, H, W)
            return_embedding: If True, return normalized embeddings
        
        Returns:
            If return_embedding=False: (logits, embeddings)
            If return_embedding=True: normalized_embeddings
        """
        B = x.shape[0]
        
        # Extract patches: (B, 3, H, W) → (B, num_patches, patch_dim)
        patches = self._extract_patches(x)  # (B, 550, 1536)
        
        # Patch embedding: (B, num_patches, patch_dim) → (B, num_patches, hidden_dim)
        patch_embeddings = self.patch_embedding(patches)  # (B, 550, 768)
        
        # Add CLS token: (B, num_patches, hidden_dim) → (B, num_patches+1, hidden_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, 768)
        embeddings = torch.cat([cls_tokens, patch_embeddings], dim=1)  # (B, 551, 768)
        
        # Add positional embeddings
        embeddings = embeddings + self.pos_embedding  # (B, 551, 768)
        
        # Transformer: (B, num_patches+1, hidden_dim) → (B, num_patches+1, hidden_dim)
        encoded = self.transformer(embeddings)  # (B, 551, 768)
        
        # Layer norm
        encoded = self.norm(encoded)
        
        # Extract CLS token
        cls_output = encoded[:, 0]  # (B, 768)
        
        # Embedding head: (B, hidden_dim) → (B, embedding_dim)
        embedding = self.embedding(cls_output)  # (B, 768)
        
        if return_embedding:
            # L2-normalize for retrieval
            return F.normalize(embedding, p=2, dim=1)
        
        # Classifier: (B, embedding_dim) → (B, num_classes)
        logits = self.classifier(embedding)
        
        return logits, embedding
    
    def _extract_patches(self, x):
        """
        Extract rectangular patches from images.
        
        Args:
            x: Images (B, 3, H, W)
        
        Returns:
            Patches (B, num_patches, patch_dim)
        """
        B, C, H, W = x.shape
        
        # Reshape to patches
        # (B, 3, 160, 1760) → (B, 3, 10, 16, 55, 32)
        x = x.reshape(B, C, self.num_patches_h, self.patch_height, 
                     self.num_patches_w, self.patch_width)
        
        # Permute: (B, 3, 10, 16, 55, 32) → (B, 10, 55, 16, 32, 3)
        x = x.permute(0, 2, 4, 3, 5, 1)
        
        # Flatten patches: (B, 10, 55, 16, 32, 3) → (B, 550, 1536)
        x = x.reshape(B, self.num_patches, -1)
        
        return x


# ============================================================
# MODEL FACTORY
# ============================================================

def create_model(config):
    """
    Create model based on configuration.
    
    Args:
        config: Configuration object with model parameters
        rank: Process rank (for printing)
    
    Returns:
        Model instance
    """
    print(f"\n{'='*70}")
    print(f"CREATING MODEL: {config.MODEL_TYPE.upper()}")
    print(f"{'='*70}")
    
    if config.MODEL_TYPE == 'convnext':
        model = ConvNeXtWriterID(
            num_classes=config.NUM_CLASSES,
            embedding_dim=config.EMBEDDING_DIM,
            dropout_embedding=config.DROPOUT_EMBEDDING,
            dropout_classifier=config.DROPOUT_CLASSIFIER,
            freeze_backbone=config.FREEZE_BACKBONE,
            freeze_layers=config.FREEZE_LAYERS,
            pretrained_path=getattr(config, 'PRETRAINED_CONVNEXT_PATH', None)
        )
    
    elif config.MODEL_TYPE == 'vit':
        model = VisionTransformerWriterID(
            num_classes=config.NUM_CLASSES,
            img_height=config.IMG_HEIGHT,
            img_width=config.IMG_WIDTH,
            patch_height=config.VIT_PATCH_HEIGHT,
            patch_width=config.VIT_PATCH_WIDTH,
            hidden_dim=config.VIT_HIDDEN_DIM,
            num_heads=config.VIT_NUM_HEADS,
            num_layers=config.VIT_NUM_LAYERS,
            embedding_dim=config.EMBEDDING_DIM,
            dropout_embedding=config.DROPOUT_EMBEDDING,
            dropout_classifier=config.DROPOUT_CLASSIFIER,
            dropout_attention=config.DROPOUT_ATTENTION,
            freeze_backbone=config.FREEZE_BACKBONE,
            freeze_layers=config.FREEZE_LAYERS
        )
    
    else:
        raise ValueError(f"Unknown model type: {config.MODEL_TYPE}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\n📊 Model Statistics:")
    print(f"  Total parameters:      {total_params:,}")
    print(f"  Trainable parameters:  {trainable_params:,}")
    print(f"  Frozen parameters:     {frozen_params:,}")
    print(f"  Embedding dimension:   {config.EMBEDDING_DIM}")
    print(f"  Number of classes:     {config.NUM_CLASSES}")
    print(f"{'='*70}\n")
    
    return model


print("✅ Section 4: Model architectures complete")


# ============================================================
# SECTION 5: LOSS FUNCTIONS
# ============================================================

# ============================================================
# FOCAL LOSS
# ============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focal Loss = -alpha * (1 - pt)^gamma * log(pt)
    
    where pt is the probability of the true class.
    
    Features:
    - Focuses on hard examples (low probability predictions)
    - Reduces loss for well-classified examples
    - Helps with class imbalance
    
    Args:
        alpha: Weighting factor (default: 1.0)
        gamma: Focusing parameter (default: 2.0)
        label_smoothing: Label smoothing factor (default: 0.0)
    
    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits (B, num_classes)
            targets: Target labels (B,)
        
        Returns:
            Focal loss (scalar)
        """
        # Compute cross-entropy with label smoothing
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', 
                                  label_smoothing=self.label_smoothing)
        
        # Compute pt = probability of true class
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


# ============================================================
# MIXUP
# ============================================================

def mixup_data(x, y, alpha=0.2):
    """
    Apply MixUp augmentation.
    
    MixUp creates virtual training examples by mixing pairs of examples.
    
    Args:
        x: Input images (B, C, H, W)
        y: Target labels (B,)
        alpha: Beta distribution parameter (default: 0.2)
    
    Returns:
        mixed_x: Mixed images
        y_a: Original labels
        y_b: Shuffled labels
        lam: Mixing coefficient
    
    Reference:
        Zhang et al. "mixup: Beyond Empirical Risk Minimization" (ICLR 2018)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute loss for MixUp.
    
    Loss = lam * loss(pred, y_a) + (1 - lam) * loss(pred, y_b)
    
    Args:
        criterion: Loss function
        pred: Predictions
        y_a: Original labels
        y_b: Shuffled labels
        lam: Mixing coefficient
    
    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# TRIPLET LOSS
# ============================================================

class TripletLoss(nn.Module):
    """
    Triplet Loss for metric learning.
    
    Encourages embeddings from same writer to be close,
    and embeddings from different writers to be far apart.
    
    Loss = max(d(anchor, positive) - d(anchor, negative) + margin, 0)
    
    Features:
    - Online hard triplet mining (within batch)
    - Hardest positive: furthest same-class example
    - Hardest negative: closest different-class example
    
    Args:
        margin: Margin for triplet loss (default: 0.5)
    
    Reference:
        Schroff et al. "FaceNet: A Unified Embedding for Face Recognition" (CVPR 2015)
    """
    
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: L2-normalized embeddings (B, embedding_dim)
            labels: Writer labels (B,)
        
        Returns:
            Triplet loss (scalar)
        """
        # Compute pairwise distances
        # (B, embedding_dim) @ (embedding_dim, B) = (B, B)
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
        
        # Create masks for same/different writers
        # mask_anchor_positive[i, j] = 1 if labels[i] == labels[j]
        mask_anchor_positive = labels.unsqueeze(1) == labels.unsqueeze(0)
        
        # mask_anchor_negative[i, j] = 1 if labels[i] != labels[j]
        mask_anchor_negative = labels.unsqueeze(1) != labels.unsqueeze(0)
        
        # ========== HARD POSITIVE MINING ==========
        # For each anchor, find the furthest positive (same writer)
        anchor_positive_dist = pairwise_dist * mask_anchor_positive.float()
        
        # Get hardest positive distance
        hardest_positive_dist, _ = anchor_positive_dist.max(dim=1)
        
        # ========== HARD NEGATIVE MINING ==========
        # For each anchor, find the closest negative (different writer)
        max_dist = pairwise_dist.max()
        anchor_negative_dist = pairwise_dist + max_dist * (~mask_anchor_negative).float()
        
        # Get hardest negative distance
        hardest_negative_dist, _ = anchor_negative_dist.min(dim=1)
        
        # ========== COMPUTE TRIPLET LOSS ==========
        triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        
        return triplet_loss.mean()


# ============================================================
# COMBINED LOSS MANAGER
# ============================================================

class LossManager:
    """
    Manages multiple losses based on configuration.
    
    Loss Strategy:
    - Option A: Focal Loss + Triplet Loss
    - Option B: CrossEntropy + MixUp + Triplet Loss
    
    Note: MixUp is ONLY used with CrossEntropy, not with Focal Loss
    
    Args:
        config: Configuration object with loss parameters
    """
    
    def __init__(self, config):
        self.config = config
        self.use_focal = config.USE_FOCAL_LOSS
        self.use_triplet = config.USE_TRIPLET_LOSS
        self.use_mixup = not config.USE_FOCAL_LOSS  # MixUp only with CE
        
        # Initialize losses
        if self.use_focal:
            self.classification_loss = FocalLoss(
                alpha=config.FOCAL_ALPHA,
                gamma=config.FOCAL_GAMMA,
                label_smoothing=config.LABEL_SMOOTHING
            )
            print(f"  Using Focal Loss (alpha={config.FOCAL_ALPHA}, gamma={config.FOCAL_GAMMA})")
        else:
            self.classification_loss = nn.CrossEntropyLoss(
                label_smoothing=config.LABEL_SMOOTHING
            )
            print(f"  Using CrossEntropy Loss (label_smoothing={config.LABEL_SMOOTHING})")
            if self.use_mixup:
                print(f"  MixUp enabled (alpha={config.MIXUP_ALPHA})")
        
        if self.use_triplet:
            self.triplet_loss = TripletLoss(margin=config.TRIPLET_MARGIN)
            self.triplet_weight = config.TRIPLET_WEIGHT
            print(f"  Triplet Loss enabled (weight={config.TRIPLET_WEIGHT}, margin={config.TRIPLET_MARGIN})")
    
    def compute_loss(self, model, images, labels, return_metrics=False):
        """
        Compute total loss based on configuration.
        
        Args:
            model: Model instance
            images: Input images (B, 3, H, W)
            labels: Target labels (B,)
            return_metrics: Whether to return individual loss components
        
        Returns:
            If return_metrics=False: total_loss
            If return_metrics=True: (total_loss, loss_dict)
        """
        loss_dict = {}
        
        # ========== MIXUP (Only with CrossEntropy) ==========
        if self.use_mixup and self.config.MIXUP_ALPHA > 0 and model.training:
            # Apply MixUp
            images, labels_a, labels_b, lam = mixup_data(
                images, labels, alpha=self.config.MIXUP_ALPHA
            )
            
            # Forward pass
            logits, embeddings = model(images)
            
            # Classification loss with MixUp
            cls_loss = mixup_criterion(
                self.classification_loss, logits, labels_a, labels_b, lam
            )
            
            loss_dict['cls_loss'] = cls_loss.item()
        else:
            # Standard forward pass (no MixUp)
            logits, embeddings = model(images)
            
            # Classification loss
            cls_loss = self.classification_loss(logits, labels)
            
            loss_dict['cls_loss'] = cls_loss.item()
        
        total_loss = cls_loss
        
        # ========== TRIPLET LOSS ==========
        if self.use_triplet:
            # Get normalized embeddings for metric learning
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Compute triplet loss
            triplet_loss = self.triplet_loss(normalized_embeddings, labels)
            
            # Add to total loss
            total_loss = total_loss + self.triplet_weight * triplet_loss
            
            loss_dict['triplet_loss'] = triplet_loss.item()
            loss_dict['triplet_weight'] = self.triplet_weight
        
        loss_dict['total_loss'] = total_loss.item()
        
        if return_metrics:
            return total_loss, loss_dict
        else:
            return total_loss
    
    def get_loss_summary(self):
        """Get summary of loss configuration."""
        summary = []
        
        if self.use_focal:
            summary.append(f"Focal(α={self.config.FOCAL_ALPHA:.2f}, γ={self.config.FOCAL_GAMMA:.1f})")
        else:
            summary.append("CrossEntropy")
            if self.use_mixup:
                summary.append(f"MixUp(α={self.config.MIXUP_ALPHA:.2f})")
        
        if self.use_triplet:
            summary.append(f"Triplet(w={self.config.TRIPLET_WEIGHT:.2f})")
        
        summary.append(f"LabelSmoothing={self.config.LABEL_SMOOTHING:.2f}")
        
        return " + ".join(summary)


# ============================================================
# LOSS TESTING
# ============================================================

def test_losses():
    """Test loss functions with dummy data."""
    print("\n" + "="*70)
    print("TESTING LOSS FUNCTIONS")
    print("="*70)
    
    # Create dummy data
    batch_size = 32
    num_classes = 7
    embedding_dim = 512
    
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    embeddings = F.normalize(torch.randn(batch_size, embedding_dim), p=2, dim=1)
    
    # Test Focal Loss
    focal = FocalLoss(alpha=1.0, gamma=2.0, label_smoothing=0.1)
    focal_loss = focal(logits, labels)
    print(f"✓ Focal Loss: {focal_loss.item():.4f}")
    
    # Test CrossEntropy
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)
    ce_loss = ce(logits, labels)
    print(f"✓ CrossEntropy Loss: {ce_loss.item():.4f}")
    
    # Test Triplet Loss
    triplet = TripletLoss(margin=0.5)
    triplet_loss = triplet(embeddings, labels)
    print(f"✓ Triplet Loss: {triplet_loss.item():.4f}")
    
    # Test MixUp
    images = torch.randn(batch_size, 3, 160, 1760)
    mixed_images, y_a, y_b, lam = mixup_data(images, labels, alpha=0.2)
    print(f"✓ MixUp: lambda={lam:.3f}")
    
    print("="*70 + "\n")


print("✅ Section 5: Loss functions complete")


# ============================================================
# SECTION 6: TRAINING & EVALUATION
# ============================================================

# ============================================================
# OPTIMIZER & SCHEDULER
# ============================================================

def create_optimizer(model, config):
    """
    Create optimizer with proper parameter grouping.
    
    Args:
        model: Model instance
        config: Configuration object
    
    Returns:
        Optimizer instance
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    return optimizer


def create_scheduler(optimizer, config, steps_per_epoch):
    """
    Create learning rate scheduler with warmup.
    
    Uses cosine annealing with linear warmup.
    
    Args:
        optimizer: Optimizer instance
        config: Configuration object
        steps_per_epoch: Number of batches per epoch
    
    Returns:
        Scheduler instance
    """
    warmup_steps = config.WARMUP_EPOCHS * steps_per_epoch
    total_steps = config.EPOCHS * steps_per_epoch
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine annealing
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return scheduler


# ============================================================
# TRAINING EPOCH
# ============================================================

def train_epoch(model, loader, loss_manager, optimizer, scheduler, scaler,
                config, epoch, rank=0):
    """
    Train for one epoch.

    Args:
        model: Model instance
        loader: Training data loader
        loss_manager: LossManager instance
        optimizer: Optimizer
        scheduler: LR scheduler
        scaler: GradScaler for AMP
        config: Configuration object
        epoch: Current epoch number
        rank: Process rank/GPU ID (default: 0 for single GPU)

    Returns:
        avg_loss: Average loss for epoch
        accuracy: Training accuracy
        loss_breakdown: Dictionary with individual loss components
    """
    model.train()
    
    running_loss = 0.0
    running_cls_loss = 0.0
    running_triplet_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar only on rank 0
    if rank == 0:
        pbar = tqdm(loader, desc=f"Epoch {epoch+1:03d} [Train]", ncols=120)
    else:
        pbar = loader
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(rank, non_blocking=True)
        labels = batch['writer_idx'].to(rank, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Forward pass with automatic mixed precision
        with autocast(enabled=config.USE_AMP):
            total_loss, loss_dict = loss_manager.compute_loss(
                model, images, labels, return_metrics=True
            )

        # ========== NaN/Inf Detection ==========
        # Critical: Detect NaN/Inf immediately to prevent wasted training
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"\n{'='*70}")
            print(f"❌ CRITICAL ERROR: Loss became {'NaN' if torch.isnan(total_loss) else 'Inf'}")
            print(f"{'='*70}")
            print(f"Epoch: {epoch+1}, Batch: {batch_idx+1}/{len(loader)}")
            print(f"Total Loss: {total_loss.item()}")
            print(f"Classification Loss: {loss_dict['cls_loss']:.4f}")
            if 'triplet_loss' in loss_dict:
                print(f"Triplet Loss: {loss_dict['triplet_loss']:.4f}")
            print(f"\nThis indicates training instability. Possible causes:")
            print(f"  • Learning rate too high")
            print(f"  • Gradient explosion")
            print(f"  • Numerical instability in loss function")
            print(f"{'='*70}\n")
            raise ValueError(f"Training failed: Loss became {'NaN' if torch.isnan(total_loss) else 'Inf'} at epoch {epoch+1}, batch {batch_idx+1}")

        # Backward pass
        scaler.scale(total_loss).backward()

        # ========== Gradient Monitoring & Clipping ==========
        scaler.unscale_(optimizer)

        # Compute gradient norm BEFORE clipping (for monitoring)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))

        # Check for extremely high gradients (indicates instability)
        if grad_norm > 100.0:
            if rank == 0 and batch_idx % 10 == 0:  # Print occasionally
                print(f"\n⚠️  WARNING: High gradient norm detected: {grad_norm:.2f}")
                print(f"   This may indicate training instability")
                print(f"   Gradient will be clipped to {config.GRADIENT_CLIP_NORM:.2f}\n")

        # Actually clip gradients to configured max norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP_NORM)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Statistics (only compute accuracy without MixUp)
        with torch.no_grad():
            if not (loss_manager.use_mixup and config.MIXUP_ALPHA > 0):
                logits, _ = model(images)
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        # Update running statistics
        running_loss += loss_dict['total_loss']
        running_cls_loss += loss_dict['cls_loss']
        if 'triplet_loss' in loss_dict:
            running_triplet_loss += loss_dict['triplet_loss']
        
        # Update progress bar (rank 0 only)
        if rank == 0:
            avg_loss = running_loss / (batch_idx + 1)
            avg_acc = 100. * correct / total if total > 0 else 0.0
            
            pbar_dict = {
                'loss': f'{avg_loss:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            }
            
            if total > 0:
                pbar_dict['acc'] = f'{avg_acc:.1f}%'
            
            pbar.set_postfix(pbar_dict)
    
    # Compute epoch statistics
    num_batches = len(loader)
    avg_loss = running_loss / num_batches
    avg_cls_loss = running_cls_loss / num_batches
    avg_triplet_loss = running_triplet_loss / num_batches if loss_manager.use_triplet else 0.0
    accuracy = 100. * correct / total if total > 0 else 0.0
    
    loss_breakdown = {
        'total': avg_loss,
        'classification': avg_cls_loss,
        'triplet': avg_triplet_loss
    }
    
    return avg_loss, accuracy, loss_breakdown


# ============================================================
# PAGE-LEVEL EVALUATION
# ============================================================

def evaluate_page_level(model, loader, pages_dict, device, writers, 
                        max_pages_per_writer=20, rank=0, return_predictions=False):
    """
    Evaluate at page level with balanced sampling.
    
    Strategy:
    1. Extract embeddings for all lines
    2. Group lines by page
    3. Compute page-level embeddings (mean of line embeddings)
    4. Sample up to max_pages_per_writer per writer (balanced)
    5. Compute page-to-page similarities
    6. Calculate metrics: Top-1, Top-5, mAP
    
    Args:
        model: Model instance
        loader: Validation/test data loader
        pages_dict: Dictionary mapping page_id -> [line_indices]
        device: Device (rank for DDP)
        writers: List of writer names
        max_pages_per_writer: Maximum pages per writer for evaluation
        rank: Process rank
        return_predictions: If True, return (results, y_true, y_pred) for confusion matrix
    
    Returns:
        Dictionary with evaluation metrics (and optionally predictions)
    """
    model.eval()
    
    # ========== STEP 1: Extract line embeddings ==========
    line_embeddings = []
    line_labels = []
    line_page_ids = []
    
    if rank == 0:
        print(f"\n{'─'*70}")
        print("PAGE-LEVEL EVALUATION")
        print(f"{'─'*70}")
    
    with torch.no_grad():
        if rank == 0:
            pbar = tqdm(loader, desc="Computing embeddings", ncols=100)
        else:
            pbar = loader
        
        for batch in pbar:
            images = batch['image'].to(device, non_blocking=True)
            embeddings = model(images, return_embedding=True)
            
            line_embeddings.append(embeddings.cpu())
            line_labels.extend(batch['writer_idx'].tolist())
            line_page_ids.extend(batch['page_id'])
    
    line_embeddings = torch.cat(line_embeddings, dim=0)
    
    # ========== STEP 2: Group by page ==========
    page_data = defaultdict(lambda: {'embeddings': [], 'label': None})
    
    for idx, page_id in enumerate(line_page_ids):
        page_data[page_id]['embeddings'].append(line_embeddings[idx])
        page_data[page_id]['label'] = line_labels[idx]
    
    # ========== STEP 3: Compute page embeddings ==========
    writer_pages = defaultdict(list)
    for page_id, data in page_data.items():
        writer_idx = data['label']
        writer_pages[writer_idx].append(page_id)
    
    # ========== STEP 4: Sample pages (balanced) ==========
    sampled_pages = []
    for writer_idx in writer_pages:
        pages = writer_pages[writer_idx]
        if len(pages) > max_pages_per_writer:
            pages = random.sample(pages, max_pages_per_writer)
        sampled_pages.extend(pages)
    
    # Compute page embeddings for sampled pages
    page_embeddings = []
    page_labels = []
    
    for page_id in sampled_pages:
        data = page_data[page_id]
        embs = torch.stack(data['embeddings'])
        page_emb = embs.mean(dim=0)  # Average pooling
        page_embeddings.append(page_emb)
        page_labels.append(data['label'])
    
    page_embeddings = torch.stack(page_embeddings)
    page_labels = torch.tensor(page_labels)
    
    # ========== STEP 5: Compute similarities ==========
    # Cosine similarity: (N, D) @ (D, N) = (N, N)
    similarities = torch.mm(page_embeddings, page_embeddings.t())
    
    # ========== STEP 6: Calculate metrics ==========
    num_writers = len(writers)
    per_writer_results = {}
    
    all_top1_correct = []
    all_top5_correct = []
    all_aps = []  # Average Precisions for mAP
    
    # For confusion matrix
    all_y_true = []
    all_y_pred = []
    
    for writer_idx in range(num_writers):
        writer_mask = page_labels == writer_idx
        if writer_mask.sum() == 0:
            continue
        
        writer_sims = similarities[writer_mask]
        writer_labels = page_labels[writer_mask]
        
        # Top-K predictions (exclude self-similarity)
        _, top5_indices = writer_sims.topk(6, dim=1)  # Get 6 because self is included
        top5_indices = top5_indices[:, 1:]  # Remove self
        top5_preds = page_labels[top5_indices]
        
        # Top-1 accuracy
        top1_preds = top5_preds[:, 0]
        top1_correct = (top1_preds == writer_labels).float().mean().item()
        
        # Collect for confusion matrix
        all_y_true.extend(writer_labels.tolist())
        all_y_pred.extend(top1_preds.tolist())
        
        # Top-5 accuracy
        top5_correct = (top5_preds == writer_labels.unsqueeze(1).expand_as(top5_preds)).any(dim=1).float().mean().item()
        
        # Average Precision (AP) for this writer
        writer_aps = []
        for i in range(writer_mask.sum()):
            # Get similarities for this query (excluding self)
            query_sims = writer_sims[i]
            query_sims[writer_mask] = float('-inf')  # Mask out same writer
            query_sims[i] = float('-inf')  # Mask out self
            
            # Sort by similarity
            sorted_indices = query_sims.argsort(descending=True)
            sorted_labels = page_labels[sorted_indices]
            
            # Compute AP
            relevant = (sorted_labels == writer_idx).float()
            if relevant.sum() > 0:
                precisions = torch.cumsum(relevant, dim=0) / torch.arange(1, len(relevant) + 1, device=relevant.device)
                ap = (precisions * relevant).sum() / relevant.sum()
                writer_aps.append(ap.item())
        
        mean_ap = np.mean(writer_aps) if writer_aps else 0.0
        
        per_writer_results[writer_idx] = {
            'top1_acc': top1_correct,
            'top5_acc': top5_correct,
            'map': mean_ap,
            'num_pages': writer_mask.sum().item()
        }
        
        all_top1_correct.append(top1_correct)
        all_top5_correct.append(top5_correct)
        all_aps.append(mean_ap)
    
    # ========== STEP 7: Macro averages ==========
    macro_top1 = np.mean(all_top1_correct)
    macro_top5 = np.mean(all_top5_correct)
    macro_map = np.mean(all_aps)
    
    results = {
        'macro_top1': macro_top1,
        'macro_top5': macro_top5,
        'macro_map': macro_map,
        'per_writer_results': per_writer_results,
        'total_pages': len(sampled_pages),
        'sampled_pages': sampled_pages
    }
    
    if return_predictions:
        return results, all_y_true, all_y_pred
    else:
        return results


# ============================================================
# EARLY STOPPING
# ============================================================

class EarlyStopping:
    """
    Early stopping to stop training when validation metric doesn't improve.
    
    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        mode: 'max' for metrics to maximize, 'min' for loss
    """
    
    def __init__(self, patience=15, min_delta=0.0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score, epoch):
        """
        Check if training should stop.
        
        Args:
            score: Current validation metric value
            epoch: Current epoch number
        
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'max':
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False
    
    def state_dict(self):
        """Return state for checkpointing."""
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'early_stop': self.early_stop
        }
    
    def load_state_dict(self, state_dict):
        """Load state from checkpoint."""
        self.counter = state_dict['counter']
        self.best_score = state_dict['best_score']
        self.best_epoch = state_dict['best_epoch']
        self.early_stop = state_dict['early_stop']


# ============================================================
# DISTRIBUTED TRAINING UTILITIES
# ============================================================

def reduce_metric(tensor, world_size):
    """
    Average metric across all processes.
    
    Args:
        tensor: Metric tensor
        world_size: Number of processes
    
    Returns:
        Averaged tensor
    """
    if not dist.is_initialized():
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def gather_metrics(metrics_dict, world_size):
    """
    Gather metrics from all processes to rank 0.
    
    Args:
        metrics_dict: Dictionary of metrics
        world_size: Number of processes
        rank: Process rank
    
    Returns:
        Gathered metrics (only on rank 0)
    """
    if not dist.is_initialized() or world_size == 1:
        return metrics_dict
    
    # Convert metrics to tensors
    gathered = {}
    
    for key, value in metrics_dict.items():
        if isinstance(value, (int, float)):
            tensor = torch.tensor(value, device=rank)
            reduced = reduce_metric(tensor, world_size)
            gathered[key] = reduced.item()
        elif isinstance(value, dict):
            # Recursively gather nested dicts
            gathered[key] = gather_metrics(value, world_size)
        else:
            gathered[key] = value
    
    return gathered


print("✅ Section 6: Training and evaluation functions complete")


# ============================================================
# SECTION 7: CHECKPOINT MANAGEMENT & TRIAL EXECUTION
# ============================================================

# ============================================================
# CHECKPOINT MANAGER
# ============================================================

class CheckpointManager:
    """
    Manages checkpoints and results for hyperparameter search trials.
    
    Directory structure:
    checkpoint_dir/
    ├── trial_000/
    │   ├── config.json                  # Trial configuration
    │   ├── best_model.pth               # Best model weights
    │   ├── final_model.pth              # Final model weights
    │   ├── metrics.json                 # Final metrics
    │   ├── training_history.json        # Epoch-by-epoch history
    │   ├── plots/
    │   │   ├── training_curves.png      # Loss and accuracy curves
    │   │   ├── validation_metrics.png   # Val metrics over time
    │   │   ├── per_writer_performance.png  # Per-writer results
    │   │   └── learning_rate.png        # LR schedule
    │   └── logs/
    │       └── training.log             # Detailed logs
    ├── trial_001/
    │   └── ...
    ├── summary/
    │   ├── all_trials_comparison.png    # Compare all trials
    │   ├── hyperparameter_importance.png
    │   ├── best_trials_summary.csv
    │   └── optimization_history.png
    └── best_overall/
        ├── best_model.pth               # Best model across all trials
        ├── config.json                  # Best configuration
        └── final_report.txt             # Comprehensive report
    
    Args:
        base_dir: Base directory for checkpoints
        trial_number: Trial number
    """
    
    def __init__(self, base_dir, trial_number=None):
        self.base_dir = Path(base_dir)
        self.trial_number = trial_number
        
        if trial_number is not None:
            self.trial_dir = self.base_dir / f'trial_{trial_number:03d}'
            self.trial_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            self.plots_dir = self.trial_dir / 'plots'
            self.plots_dir.mkdir(exist_ok=True)
            
            self.logs_dir = self.trial_dir / 'logs'
            self.logs_dir.mkdir(exist_ok=True)
        else:
            self.trial_dir = None
        
        # Summary directories
        self.summary_dir = self.base_dir / 'summary'
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_overall_dir = self.base_dir / 'best_overall'
        self.best_overall_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_metric = 0.0
        self.best_epoch = 0
    
    def save_config(self, config):
        """Save trial configuration."""
        config_dict = {}
        for attr in dir(config):
            if not attr.startswith('_') and attr.isupper():
                value = getattr(config, attr)
                if isinstance(value, (str, int, float, bool, list)):
                    config_dict[attr] = value
                else:
                    config_dict[attr] = str(value)
        
        config_path = self.trial_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def save_checkpoint(self, epoch, model, optimizer, scheduler, scaler,
                       metrics, history, is_best=False):
        """
        Save checkpoint.
        
        Args:
            epoch: Current epoch
            model: Model (unwrap from DDP if needed)
            optimizer: Optimizer
            scheduler: Scheduler
            scaler: GradScaler
            metrics: Current metrics dict
            history: Training history dict
            is_best: Whether this is the best model so far
        """
        # Unwrap model from DDP
        model_to_save = model.module if hasattr(model, 'module') else model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'metrics': metrics,
            'history': history,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch
        }
        
        # Save latest checkpoint
        latest_path = self.trial_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best model
        if is_best:
            best_path = self.trial_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.best_metric = metrics['val_macro_top1']
            self.best_epoch = epoch
    
    def save_final_model(self, model, metrics, history):
        """Save final model and metrics."""
        # Unwrap model from DDP
        model_to_save = model.module if hasattr(model, 'module') else model
        
        # Save final model weights
        final_path = self.trial_dir / 'final_model.pth'
        torch.save({
            'model_state_dict': model_to_save.state_dict(),
            'metrics': metrics,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch
        }, final_path)
        
        # Save metrics
        metrics_path = self.trial_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save training history
        history_path = self.trial_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    def save_training_plots(self, history, epoch):
        """
        Save training visualization plots.
        
        Args:
            history: Training history dict
            epoch: Current epoch
        """
        if len(history['train_loss']) == 0:
            return
        
        epochs_range = range(1, len(history['train_loss']) + 1)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Training Loss
        axes[0, 0].plot(epochs_range, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # 2. Training Accuracy
        if 'train_acc' in history and len(history['train_acc']) > 0:
            axes[0, 1].plot(epochs_range, history['train_acc'], 'g-', label='Train Acc', linewidth=2)
            axes[0, 1].set_xlabel('Epoch', fontsize=12)
            axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
            axes[0, 1].set_title('Training Accuracy', fontsize=14, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
        
        # 3. Validation Top-1 and Top-5
        axes[1, 0].plot(epochs_range, [x*100 for x in history['val_macro_top1']], 
                       'r-', label='Val Top-1', linewidth=2)
        if 'val_macro_top5' in history:
            axes[1, 0].plot(epochs_range, [x*100 for x in history['val_macro_top5']], 
                           'orange', label='Val Top-5', linewidth=2, linestyle='--')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1, 0].set_title('Validation Accuracy (Macro)', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Mark best epoch
        best_idx = np.argmax(history['val_macro_top1'])
        best_val = history['val_macro_top1'][best_idx] * 100
        axes[1, 0].axvline(x=best_idx+1, color='red', linestyle=':', alpha=0.5)
        axes[1, 0].text(best_idx+1, best_val, f'Best\n{best_val:.1f}%', 
                       ha='left', va='bottom', fontsize=10, color='red')
        
        # 4. Validation mAP
        if 'val_macro_map' in history and len(history['val_macro_map']) > 0:
            axes[1, 1].plot(epochs_range, history['val_macro_map'], 
                           'purple', label='Val mAP', linewidth=2)
            axes[1, 1].set_xlabel('Epoch', fontsize=12)
            axes[1, 1].set_ylabel('mAP', fontsize=12)
            axes[1, 1].set_title('Validation mAP (Macro)', fontsize=14, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_per_writer_plot(self, per_writer_results, writers):
        """
        Save per-writer performance plot.
        
        Args:
            per_writer_results: Dict of per-writer metrics
            writers: List of writer names
        """
        if not per_writer_results:
            return
        
        # Prepare data
        writer_names = []
        top1_accs = []
        top5_accs = []
        maps = []
        
        for writer_idx, metrics in sorted(per_writer_results.items()):
            writer_names.append(writers[writer_idx])
            top1_accs.append(metrics['top1_acc'] * 100)
            top5_accs.append(metrics['top5_acc'] * 100)
            maps.append(metrics['map'])
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        x = np.arange(len(writer_names))
        width = 0.35
        
        # Top-1 Accuracy
        axes[0].bar(x, top1_accs, color='steelblue', alpha=0.8)
        axes[0].set_xlabel('Writer', fontsize=12)
        axes[0].set_ylabel('Top-1 Accuracy (%)', fontsize=12)
        axes[0].set_title('Per-Writer Top-1 Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(writer_names, rotation=45, ha='right')
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].axhline(y=np.mean(top1_accs), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(top1_accs):.1f}%')
        axes[0].legend()
        
        # Top-5 Accuracy
        axes[1].bar(x, top5_accs, color='coral', alpha=0.8)
        axes[1].set_xlabel('Writer', fontsize=12)
        axes[1].set_ylabel('Top-5 Accuracy (%)', fontsize=12)
        axes[1].set_title('Per-Writer Top-5 Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(writer_names, rotation=45, ha='right')
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].axhline(y=np.mean(top5_accs), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(top5_accs):.1f}%')
        axes[1].legend()
        
        # mAP
        axes[2].bar(x, maps, color='purple', alpha=0.8)
        axes[2].set_xlabel('Writer', fontsize=12)
        axes[2].set_ylabel('mAP', fontsize=12)
        axes[2].set_title('Per-Writer mAP', fontsize=14, fontweight='bold')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(writer_names, rotation=45, ha='right')
        axes[2].grid(axis='y', alpha=0.3)
        axes[2].axhline(y=np.mean(maps), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(maps):.3f}')
        axes[2].legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / 'per_writer_performance.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_learning_rate_plot(self, lr_history):
        """Save learning rate schedule plot."""
        if not lr_history:
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(lr_history, linewidth=2, color='blue')
        plt.xlabel('Step', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Save plot
        plot_path = self.plots_dir / 'learning_rate.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_confusion_matrix(self, y_true, y_pred, writers, normalize=True):
        """
        Save confusion matrix plot.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            writers: List of writer names
            normalize: Whether to normalize (show percentages)
        """
        from sklearn.metrics import confusion_matrix
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix (%)'
        else:
            fmt = 'd'
            title = 'Confusion Matrix (Counts)'
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 12))
        
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        # Set ticks
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=writers,
               yticklabels=writers,
               title=title,
               ylabel='True Writer',
               xlabel='Predicted Writer')
        
        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if normalize:
                    text_val = f'{cm[i, j]*100:.1f}' if cm[i, j] > 0.01 else ''
                else:
                    text_val = f'{cm[i, j]:.0f}' if cm[i, j] > 0 else ''
                
                ax.text(j, i, text_val,
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black",
                       fontsize=7)
        
        plt.tight_layout()
        
        # Save plot
        suffix = 'normalized' if normalize else 'counts'
        plot_path = self.plots_dir / f'confusion_matrix_{suffix}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also save as numpy array
        np.save(self.plots_dir / 'confusion_matrix.npy', cm)
    
    def save_trial_summary(self, config, metrics, history):
        """
        Save comprehensive trial summary.

        Args:
            config: Trial configuration
            metrics: Final metrics
            history: Training history
        """
        summary_path = self.trial_dir / 'trial_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"TRIAL {self.trial_number} SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            # Configuration
            f.write("CONFIGURATION:\n")
            f.write("-"*70 + "\n")
            f.write(f"Model Type:           {config.MODEL_TYPE}\n")
            f.write(f"Learning Rate:        {config.LEARNING_RATE:.2e}\n")
            f.write(f"Weight Decay:         {config.WEIGHT_DECAY:.2e}\n")
            f.write(f"Batch Size:           {config.BATCH_SIZE}\n")
            f.write(f"Embedding Dim:        {config.EMBEDDING_DIM}\n")
            f.write(f"Dropout (Emb):        {config.DROPOUT_EMBEDDING}\n")
            f.write(f"Dropout (Clf):        {config.DROPOUT_CLASSIFIER}\n")
            f.write(f"Label Smoothing:      {config.LABEL_SMOOTHING}\n")
            f.write(f"Freeze Backbone:      {config.FREEZE_BACKBONE}\n")
            if config.FREEZE_BACKBONE:
                f.write(f"Freeze Layers:        {config.FREEZE_LAYERS}\n")
            f.write(f"\nLoss Configuration:\n")
            f.write(f"Use Focal Loss:       {config.USE_FOCAL_LOSS}\n")
            if config.USE_FOCAL_LOSS:
                f.write(f"  Focal Gamma:        {config.FOCAL_GAMMA}\n")
                f.write(f"  Focal Alpha:        {config.FOCAL_ALPHA}\n")
            else:
                f.write(f"  MixUp Alpha:        {config.MIXUP_ALPHA}\n")
            f.write(f"Use Triplet Loss:     {config.USE_TRIPLET_LOSS}\n")
            if config.USE_TRIPLET_LOSS:
                f.write(f"  Triplet Weight:     {config.TRIPLET_WEIGHT}\n")
                f.write(f"  Triplet Margin:     {config.TRIPLET_MARGIN}\n")
            f.write(f"\nSampler Configuration:\n")
            f.write(f"Quota per Writer:     300\n")
            f.write(f"Max Exposure:         2\n")
            
            # Results
            f.write(f"\n{'='*70}\n")
            f.write("RESULTS:\n")
            f.write("-"*70 + "\n")
            f.write(f"Best Epoch:           {self.best_epoch + 1}\n")
            f.write(f"Total Epochs:         {len(history['train_loss'])}\n")
            f.write(f"\nValidation Metrics (Best Epoch):\n")
            f.write(f"  Macro Top-1:        {metrics['val_macro_top1']*100:.2f}%\n")
            f.write(f"  Macro Top-5:        {metrics['val_macro_top5']*100:.2f}%\n")
            f.write(f"  Macro mAP:          {metrics['val_macro_map']:.4f}\n")
            f.write(f"\nFinal Training Loss:  {history['train_loss'][-1]:.4f}\n")
            
            # Per-writer results
            if 'per_writer_results' in metrics:
                f.write(f"\n{'='*70}\n")
                f.write("PER-WRITER RESULTS:\n")
                f.write("-"*70 + "\n")
                f.write(f"{'Writer':<25} {'Top-1':>8} {'Top-5':>8} {'mAP':>8} {'Pages':>8}\n")
                f.write("-"*70 + "\n")
                
                for writer_idx, results in sorted(metrics['per_writer_results'].items()):
                    from pathlib import Path
                    # Get writer name from config
                    writers_list = config.WRITERS if hasattr(config, 'WRITERS') else []
                    writer_name = writers_list[writer_idx] if writer_idx < len(writers_list) else f"Writer_{writer_idx}"
                    
                    f.write(f"{writer_name:<25} "
                           f"{results['top1_acc']*100:>7.1f}% "
                           f"{results['top5_acc']*100:>7.1f}% "
                           f"{results['map']:>7.3f} "
                           f"{results['num_pages']:>8d}\n")
            
            f.write("\n" + "="*70 + "\n")


# ============================================================
# TRIAL RUNNER
# ============================================================

def run_trial(trial, config_base, data_index):
    """
    Run a single hyperparameter search trial.

    Args:
        trial: Optuna trial object
        config_base: Base configuration
        data_index: Pre-built data index (writers, writer2idx, split_data)

    Returns:
        Best validation macro top-1 accuracy
    """
    try:
        # Suggest hyperparameters
        hyperparams = HyperparameterSpace.suggest_hyperparameters(trial, quick_test=config_base.QUICK_TEST)
    except Exception as e:
        print(f"\n❌ Trial {trial.number} failed during hyperparameter suggestion: {e}")
        import traceback
        traceback.print_exc()
        raise

    try:
        # Create trial configuration
        config = TrialConfig(hyperparams)
    
        # Copy base config attributes
        config.DATA_ROOT = config_base.DATA_ROOT
        config.CHECKPOINT_DIR = config_base.CHECKPOINT_DIR
        config.WRITERS = data_index[0]  # Writer names
        config.NUM_CLASSES = len(config.WRITERS)

        # ========== Validate Configuration ==========
        # Ensure batch size is reasonable (prevent tiny batches that cause issues)
        if config.BATCH_SIZE < 16:
            raise ValueError(f"Batch size {config.BATCH_SIZE} is too small! Minimum is 16. "
                           f"Small batches can cause training instability and poor performance.")

        # Print trial info
        print("\n" + "="*70)
        print(f"TRIAL {trial.number}")
        print("="*70)
        print(f"Model: {config.MODEL_TYPE}")
        print(f"LR: {config.LEARNING_RATE:.2e}, WD: {config.WEIGHT_DECAY:.2e}")
        print(f"Dropout: Emb={config.DROPOUT_EMBEDDING}, Clf={config.DROPOUT_CLASSIFIER}")
        print(f"Loss: {'Focal' if config.USE_FOCAL_LOSS else 'CE+MixUp'} + {'Triplet' if config.USE_TRIPLET_LOSS else 'No Triplet'}")
        print("="*70 + "\n")

        # Initialize checkpoint manager
        checkpoint_manager = CheckpointManager(
            base_dir=config.CHECKPOINT_DIR,
            trial_number=trial.number
        )

        checkpoint_manager.save_config(config)
    
        # Unpack data index
        writers, writer2idx, split_data = data_index
    
        # Create datasets
        train_dataset = ArabicLineDataset(
            split_data['train']['lines'],
            config,
            is_train=True
        )
    
        val_dataset = ArabicLineDataset(
            split_data['val']['lines'],
            config,
            is_train=False
        )
    
        # Create sampler
        # Use fewer lines per writer in quick test mode
        quota_per_writer = 50 if config_base.QUICK_TEST else 300
        train_sampler = WriterBalancedSampler(
            split_data['train']['lines'],
            quota_per_writer=quota_per_writer,
            max_exposure=2
        )

        train_sampler.print_sampler_info(writers)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            sampler=train_sampler,
            num_workers=config.NUM_WORKERS,
            pin_memory=True
        )
    
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True
        )
    
        # Create model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rank = 0  # Single GPU mode (always rank 0, used for device and logging)
        model = create_model(config)
        model = model.to(device)
    
        # Create loss manager
        loss_manager = LossManager(config)
    
        # Create optimizer and scheduler
        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(optimizer, config, len(train_loader))
        scaler = GradScaler(enabled=config.USE_AMP)
    
        # Early stopping
        early_stopping = EarlyStopping(
            patience=config.EARLY_STOPPING_PATIENCE,
            min_delta=config.EARLY_STOPPING_MIN_DELTA,
            mode='max'
        )
    
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_macro_top1': [],
            'val_macro_top5': [],
            'val_macro_map': [],
            'lr': []
        }

        best_val_top1 = 0.0
        best_per_writer_results = None  # Will be set when we find best epoch

        # Training loop
        for epoch in range(config.EPOCHS):
            # Train
            train_loss, train_acc, loss_breakdown = train_epoch(
                model, train_loader, loss_manager, optimizer, scheduler, scaler,
                config, epoch, rank=rank
            )

            # ========== Sanity Check: Detect NaN in returned loss ==========
            if np.isnan(train_loss) or np.isinf(train_loss):
                if rank == 0:
                    print(f"\n{'='*70}")
                    print(f"❌ TRAINING FAILED: Epoch {epoch+1} returned {'NaN' if np.isnan(train_loss) else 'Inf'} loss")
                    print(f"{'='*70}")
                    print(f"Training will be stopped to prevent wasted computation.")
                    print(f"This trial will be marked as failed.\n")
                raise ValueError(f"Training failed at epoch {epoch+1}: loss is {'NaN' if np.isnan(train_loss) else 'Inf'}")

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['lr'].append(scheduler.get_last_lr()[0])
        
            # Evaluate
            val_results = evaluate_page_level(
                model, val_loader,
                split_data['val']['pages'],
                rank, writers,
                max_pages_per_writer=config.MAX_PAGES_PER_WRITER_VAL,
                rank=rank,
                return_predictions=True  # Get predictions for confusion matrix
            )
        
            # Unpack results
            if isinstance(val_results, tuple):
                val_metrics, y_true, y_pred = val_results
            else:
                val_metrics = val_results
                y_true, y_pred = None, None
        
            history['val_macro_top1'].append(val_metrics['macro_top1'])
            history['val_macro_top5'].append(val_metrics['macro_top5'])
            history['val_macro_map'].append(val_metrics['macro_map'])
        
            # Print results (rank 0 only)
            if rank == 0:
                print(f"\n{'─'*70}")
                print(f"Epoch {epoch+1}/{config.EPOCHS} Results:")
                print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                print(f"  Val Top-1: {val_metrics['macro_top1']*100:.2f}% | "
                      f"Top-5: {val_metrics['macro_top5']*100:.2f}% | "
                      f"mAP: {val_metrics['macro_map']:.4f}")
                print(f"{'─'*70}")
        
            # Check if best
            is_best = val_metrics['macro_top1'] > best_val_top1
            if is_best:
                best_val_top1 = val_metrics['macro_top1']
                best_y_true = y_true
                best_y_pred = y_pred
                best_per_writer_results = val_metrics['per_writer_results']  # Save best per-writer results

            # Save checkpoint
            if rank == 0:
                metrics = {
                    'val_macro_top1': val_metrics['macro_top1'],
                    'val_macro_top5': val_metrics['macro_top5'],
                    'val_macro_map': val_metrics['macro_map'],
                    'per_writer_results': val_metrics['per_writer_results']
                }
            
                checkpoint_manager.save_checkpoint(
                    epoch, model, optimizer, scheduler, scaler,
                    metrics, history, is_best=is_best
                )
            
                # Save plots periodically
                if (epoch + 1) % 5 == 0 or is_best:
                    checkpoint_manager.save_training_plots(history, epoch)
                
                    # Save confusion matrix for best epoch
                    if is_best and y_true is not None and y_pred is not None:
                        checkpoint_manager.save_confusion_matrix(y_true, y_pred, writers, normalize=True)
                        checkpoint_manager.save_confusion_matrix(y_true, y_pred, writers, normalize=False)
        
            # Report to Optuna (for pruning)
            trial.report(val_metrics['macro_top1'], epoch)

            # Check if trial should be pruned
            if trial.should_prune():
                if rank == 0:
                    print(f"\n⚠️  Trial {trial.number} pruned at epoch {epoch+1}")
                raise optuna.TrialPruned()

            # Early stopping
            if early_stopping(val_metrics['macro_top1'], epoch):
                if rank == 0:
                    print(f"\n🛑 Early stopping at epoch {epoch+1}")
                break
    
        # Save final results (rank 0 only)
        if rank == 0:
            final_metrics = {
                'val_macro_top1': best_val_top1,
                'val_macro_top5': history['val_macro_top5'][np.argmax(history['val_macro_top1'])],
                'val_macro_map': history['val_macro_map'][np.argmax(history['val_macro_top1'])],
                'per_writer_results': best_per_writer_results
            }

            checkpoint_manager.save_final_model(model, final_metrics, history)
            checkpoint_manager.save_training_plots(history, len(history['train_loss'])-1)
            checkpoint_manager.save_per_writer_plot(best_per_writer_results, writers)
            checkpoint_manager.save_learning_rate_plot(history['lr'])
            checkpoint_manager.save_trial_summary(config, final_metrics, history)

            # Print training coverage report
            train_sampler.print_coverage_report(writers)

            print(f"\n✅ Trial {trial.number} completed | Best Val Top-1: {best_val_top1*100:.2f}%\n")

        return best_val_top1

    except optuna.TrialPruned:
        # Trial was pruned - this is expected, re-raise
        if rank == 0:
            print(f"\n⚠️  Trial {trial.number} was pruned (early stopping by Optuna)")
        raise

    except Exception as e:
        # Unexpected error - log and fail the trial
        if rank == 0:
            print(f"\n❌ Trial {trial.number} FAILED with error: {e}")
            import traceback
            traceback.print_exc()

            # Save error log to trial directory
            try:
                error_log_path = Path(config_base.CHECKPOINT_DIR) / f'trial_{trial.number:03d}' / 'error.log'
                error_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(error_log_path, 'w') as f:
                    f.write(f"Trial {trial.number} failed at {datetime.now()}\n")
                    f.write(f"Error: {e}\n\n")
                    f.write("Traceback:\n")
                    traceback.print_exc(file=f)
                print(f"   Error log saved to: {error_log_path}")
            except:
                pass

        # Re-raise the exception so Optuna marks this trial as failed
        raise


print("✅ Section 7: Checkpoint management and trial execution complete")


# ============================================================
# SECTION 8: MAIN EXECUTION & SUMMARY ANALYSIS
# ============================================================

# ============================================================
# SUMMARY ANALYSIS
# ============================================================

def create_summary_plots(study, checkpoint_dir):
    """
    Create comprehensive summary plots across all trials.
    
    Args:
        study: Optuna study object
        checkpoint_dir: Base checkpoint directory
    """
    summary_dir = Path(checkpoint_dir) / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    # Get completed trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if len(completed_trials) == 0:
        print("⚠️  No completed trials to analyze")
        return
    
    print(f"\n{'='*70}")
    print(f"CREATING SUMMARY ANALYSIS ({len(completed_trials)} trials)")
    print(f"{'='*70}\n")
    
    # ========== 1. OPTIMIZATION HISTORY ==========
    plt.figure(figsize=(12, 6))
    
    trial_numbers = [t.number for t in completed_trials]
    values = [t.value for t in completed_trials]
    best_values = [max(values[:i+1]) for i in range(len(values))]
    
    plt.plot(trial_numbers, values, 'o-', label='Trial Value', alpha=0.6)
    plt.plot(trial_numbers, best_values, 'r-', linewidth=2, label='Best So Far')
    plt.xlabel('Trial Number', fontsize=12)
    plt.ylabel('Validation Macro Top-1', fontsize=12)
    plt.title('Hyperparameter Search Progress', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Mark best trial
    best_idx = np.argmax(values)
    plt.scatter([trial_numbers[best_idx]], [values[best_idx]], 
               color='gold', s=200, marker='*', zorder=5, 
               label=f'Best: {values[best_idx]*100:.2f}%')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(summary_dir / 'optimization_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ========== 2. PARAMETER IMPORTANCE ==========
    try:
        importance = optuna.importance.get_param_importances(study)
        
        if importance:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            params = list(importance.keys())
            importances = list(importance.values())
            
            # Sort by importance
            sorted_indices = np.argsort(importances)
            params = [params[i] for i in sorted_indices]
            importances = [importances[i] for i in sorted_indices]
            
            ax.barh(params, importances, color='steelblue', alpha=0.8)
            ax.set_xlabel('Importance', fontsize=12)
            ax.set_title('Hyperparameter Importance', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(summary_dir / 'hyperparameter_importance.png', dpi=150, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"  ⚠️  Could not compute parameter importance: {e}")
    
    # ========== 3. PARALLEL COORDINATE PLOT ==========
    try:
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_image(str(summary_dir / 'parallel_coordinates.png'))
    except Exception as e:
        print(f"  ⚠️  Could not create parallel coordinate plot: {e}")
    
    # ========== 4. HYPERPARAMETER RELATIONSHIPS ==========
    # Model type vs Performance
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract hyperparameters
    model_types = []
    lrs = []
    wds = []
    dropout_embs = []
    values_list = []
    
    for trial in completed_trials:
        model_types.append(trial.params.get('model_type', 'unknown'))
        lrs.append(trial.params.get('learning_rate', 0))
        wds.append(trial.params.get('weight_decay', 0))
        dropout_embs.append(trial.params.get('dropout_embedding', 0))
        values_list.append(trial.value * 100)
    
    # Plot 1: Model Type
    model_perf = defaultdict(list)
    for mt, val in zip(model_types, values_list):
        model_perf[mt].append(val)
    
    axes[0, 0].boxplot([model_perf[mt] for mt in model_perf.keys()], 
                       labels=list(model_perf.keys()))
    axes[0, 0].set_ylabel('Val Top-1 (%)', fontsize=11)
    axes[0, 0].set_title('Performance by Model Type', fontsize=12, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Learning Rate
    axes[0, 1].scatter(lrs, values_list, alpha=0.6, s=80, c=values_list, cmap='viridis')
    axes[0, 1].set_xlabel('Learning Rate', fontsize=11)
    axes[0, 1].set_ylabel('Val Top-1 (%)', fontsize=11)
    axes[0, 1].set_title('Learning Rate vs Performance', fontsize=12, fontweight='bold')
    axes[0, 1].set_xscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Weight Decay
    axes[1, 0].scatter(wds, values_list, alpha=0.6, s=80, c=values_list, cmap='viridis')
    axes[1, 0].set_xlabel('Weight Decay', fontsize=11)
    axes[1, 0].set_ylabel('Val Top-1 (%)', fontsize=11)
    axes[1, 0].set_title('Weight Decay vs Performance', fontsize=12, fontweight='bold')
    axes[1, 0].set_xscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Dropout
    axes[1, 1].scatter(dropout_embs, values_list, alpha=0.6, s=80, c=values_list, cmap='viridis')
    axes[1, 1].set_xlabel('Dropout (Embedding)', fontsize=11)
    axes[1, 1].set_ylabel('Val Top-1 (%)', fontsize=11)
    axes[1, 1].set_title('Dropout vs Performance', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(summary_dir / 'hyperparameter_relationships.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Summary plots created")


def create_trials_comparison_table(study, checkpoint_dir):
    """
    Create CSV table comparing all trials.
    
    Args:
        study: Optuna study object
        checkpoint_dir: Base checkpoint directory
    """
    summary_dir = Path(checkpoint_dir) / 'summary'
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if len(completed_trials) == 0:
        return
    
    # Collect trial data
    rows = []
    for trial in completed_trials:
        row = {
            'trial_number': trial.number,
            'val_top1': trial.value * 100,
            'model_type': trial.params.get('model_type', ''),
            'learning_rate': trial.params.get('learning_rate', 0),
            'weight_decay': trial.params.get('weight_decay', 0),
            'batch_size': trial.params.get('batch_size', 0),
            'dropout_embedding': trial.params.get('dropout_embedding', 0),
            'dropout_classifier': trial.params.get('dropout_classifier', 0),
            'label_smoothing': trial.params.get('label_smoothing', 0),
            'use_focal': trial.params.get('use_focal_loss', False),
            'use_triplet': trial.params.get('use_triplet_loss', False),
            'freeze_backbone': trial.params.get('freeze_backbone', False),
            'embedding_dim': trial.params.get('embedding_dim', 0)
        }
        rows.append(row)
    
    # Sort by performance
    rows = sorted(rows, key=lambda x: x['val_top1'], reverse=True)
    
    # Save to CSV
    import csv
    csv_path = summary_dir / 'all_trials_comparison.csv'
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✅ Trials comparison saved to: {csv_path}")
    
    # Also save top 5 trials
    top5_path = summary_dir / 'top5_trials.csv'
    with open(top5_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows[:5])
    
    print(f"✅ Top 5 trials saved to: {top5_path}")


def save_best_model_globally(study, checkpoint_dir):
    """
    Copy best model to best_overall directory.
    
    Args:
        study: Optuna study object
        checkpoint_dir: Base checkpoint directory
    """
    best_trial = study.best_trial
    best_trial_dir = Path(checkpoint_dir) / f'trial_{best_trial.number:03d}'
    best_overall_dir = Path(checkpoint_dir) / 'best_overall'
    best_overall_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy best model
    import shutil
    
    src_model = best_trial_dir / 'best_model.pth'
    dst_model = best_overall_dir / 'best_model.pth'
    if src_model.exists():
        shutil.copy2(src_model, dst_model)
    
    # Copy config
    src_config = best_trial_dir / 'config.json'
    dst_config = best_overall_dir / 'config.json'
    if src_config.exists():
        shutil.copy2(src_config, dst_config)
    
    # Copy metrics
    src_metrics = best_trial_dir / 'metrics.json'
    dst_metrics = best_overall_dir / 'metrics.json'
    if src_metrics.exists():
        shutil.copy2(src_metrics, dst_metrics)
    
    # Copy plots
    src_plots = best_trial_dir / 'plots'
    dst_plots = best_overall_dir / 'plots'
    if src_plots.exists():
        if dst_plots.exists():
            shutil.rmtree(dst_plots)
        shutil.copytree(src_plots, dst_plots)
    
    # Create final report
    report_path = best_overall_dir / 'final_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("HYPERPARAMETER SEARCH - FINAL REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total Trials: {len(study.trials)}\n")
        f.write(f"Completed Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}\n")
        f.write(f"Pruned Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}\n\n")
        
        f.write("="*70 + "\n")
        f.write("BEST TRIAL CONFIGURATION\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Trial Number: {best_trial.number}\n")
        f.write(f"Validation Macro Top-1: {best_trial.value*100:.2f}%\n\n")
        
        f.write("Hyperparameters:\n")
        f.write("-"*70 + "\n")
        for key, value in sorted(best_trial.params.items()):
            f.write(f"  {key:.<40} {value}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("RECOMMENDATION\n")
        f.write("="*70 + "\n\n")
        f.write(f"Use the configuration from trial_{best_trial.number:03d} for final training\n")
        f.write(f"on combined train+val split.\n\n")
        f.write(f"Expected performance on test set: ~{best_trial.value*100:.1f}% (±2-3%)\n\n")
        
        f.write("="*70 + "\n")
    
    print(f"\n✅ Best model saved to: {best_overall_dir}")
    print(f"✅ Final report: {report_path}")


# ============================================================
# HYPERPARAMETER SEARCH RUNNER
# ============================================================

def run_hyperparameter_search(config_base, n_trials):
    """
    Run hyperparameter search on single GPU.

    Args:
        config_base: Base configuration
        n_trials: Number of trials to run
    """
    # Build data index
    print("Building data index...")

    # Determine which writers to use
    if config_base.USE_ALL_WRITERS:
        selected_writers = None  # Use all writers
    else:
        # Use subset: discover writers first, then select first N
        all_writers = discover_writers(config_base.DATA_ROOT, split='train')
        selected_writers = all_writers[:config_base.NUM_WRITERS_SUBSET]
        print(f"\n✅ Using subset: {config_base.NUM_WRITERS_SUBSET} writers")
        for i, w in enumerate(selected_writers, 1):
            print(f"  {i}. {w}")

    writers, writer2idx, split_data = build_index(
        config_base.DATA_ROOT,
        splits=['train', 'val'],
        selected_writers=selected_writers
    )
    data_index = (writers, writer2idx, split_data)

    HyperparameterSpace.print_search_space_summary()

    # Ensure checkpoint directory exists
    checkpoint_dir = Path(config_base.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create SQLite storage for persistence (resumability)
    storage_path = checkpoint_dir / 'optuna_study.db'
    storage = f'sqlite:///{storage_path}'
    study_name = 'arabic_writer_id_hyperparam_search'

    # Load or create study
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage
        )
        n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        n_failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])

        print(f"\n{'='*70}")
        print(f"RESUMING EXISTING STUDY")
        print(f"{'='*70}")
        print(f"Study loaded from: {storage_path}")
        print(f"Completed trials: {n_completed}")
        print(f"Pruned trials: {n_pruned}")
        print(f"Failed trials: {n_failed}")
        print(f"Remaining trials: {max(0, n_trials - n_completed - n_pruned - n_failed)}")
        if n_completed > 0:
            print(f"Best value so far: {study.best_trial.value*100:.2f}%")
        print(f"{'='*70}\n")

    except KeyError:
        # Study doesn't exist, create new one
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            load_if_exists=False
        )

        print(f"\n{'='*70}")
        print(f"STARTING NEW HYPERPARAMETER SEARCH")
        print(f"{'='*70}")
        print(f"Study database: {storage_path}")
        print(f"Number of trials: {n_trials}")
        print(f"Optimization metric: Validation Macro Top-1 Accuracy")
        print(f"Note: This search is resumable. You can safely stop and restart.")
        print(f"{'='*70}\n")

    # Run optimization
    # Calculate how many trials still needed
    n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    total_trials_attempted = len(study.trials)
    n_remaining = max(0, n_trials - total_trials_attempted)

    if n_remaining > 0:
        study.optimize(
            lambda trial: run_trial(trial, config_base, data_index),
            n_trials=n_remaining,
            show_progress_bar=True
        )
    else:
        print(f"\n✅ All {n_trials} trials already completed. Skipping optimization.")
        print(f"   Use --n_trials to run more trials.\n")
        
    # Print results
    print("\n" + "="*70)
    print("HYPERPARAMETER SEARCH COMPLETED")
    print("="*70)
    print(f"\nBest Trial: {study.best_trial.number}")
    print(f"Best Value: {study.best_trial.value*100:.2f}%")
    print(f"\nBest Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
        
    # Create summary analysis
    print("\nCreating summary analysis...")
    create_summary_plots(study, config_base.CHECKPOINT_DIR)
    create_trials_comparison_table(study, config_base.CHECKPOINT_DIR)
    save_best_model_globally(study, config_base.CHECKPOINT_DIR)
        
    print("\n" + "="*70)
    print("ALL DONE! 🎉")
    print("="*70)
    print(f"\nResults saved to: {config_base.CHECKPOINT_DIR}")
    print(f"Best model: {config_base.CHECKPOINT_DIR}/best_overall/")
    print(f"Summary: {config_base.CHECKPOINT_DIR}/summary/")
    print("="*70 + "\n")


# ============================================================
# PARALLEL MODE (MULTI-GPU)
# ============================================================

def run_parallel_search(args, num_gpus):
    """
    Run parallel hyperparameter search with one trial per GPU.

    Args:
        args: Command line arguments
        num_gpus: Number of GPUs available
    """
    print("\n" + "="*70)
    print("PARALLEL HYPERPARAMETER SEARCH")
    print("="*70)
    print(f"Available GPUs: {num_gpus}")
    print(f"Total trials: {args.n_trials}")
    print(f"Trials per GPU: ~{args.n_trials // num_gpus}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Data root: {args.data_root}")
    print("="*70)

    # Create checkpoint and log directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.checkpoint_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print("\n🚀 Launching workers...\n")

    processes = []

    # Register signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print("\n\n" + "="*70)
        print("🛑 INTERRUPT RECEIVED - Gracefully stopping all workers...")
        print("="*70)
        for i, proc in enumerate(processes):
            if proc.poll() is None:
                print(f"  Stopping worker on GPU {i}...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"  Force killing worker on GPU {i}...")
                    proc.kill()
        print("✅ All workers stopped")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Launch one worker per GPU
    for gpu_id in range(num_gpus):
        # Build command
        cmd = [
            sys.executable,
            __file__,  # This script
            "--data_root", args.data_root,
            "--checkpoint_dir", args.checkpoint_dir,
            "--n_trials", str(args.n_trials),
            "--single_gpu_mode",  # Flag to run in single GPU mode
        ]

        if args.use_all_writers:
            cmd.append("--use_all_writers")
        else:
            cmd.extend(["--num_writers_subset", str(args.num_writers_subset)])

        if args.quick_test:
            cmd.append("--quick_test")

        # Create log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"worker_gpu{gpu_id}_{timestamp}.log"

        # Set environment to restrict to single GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # Launch process
        log_handle = open(log_file, 'w')
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        processes.append(proc)
        print(f"  ✅ GPU {gpu_id}: Worker launched (PID {proc.pid})")
        print(f"     Log: {log_file}")

        # Small delay to stagger database access
        time.sleep(2)

    print("\n" + "="*70)
    print("ALL WORKERS LAUNCHED")
    print("="*70)
    print("\n📊 Monitor progress:")
    print(f"  • Check logs: {log_dir}")
    print(f"  • Check trials: {Path(args.checkpoint_dir) / 'trial_*'}")
    print(f"  • Database: {Path(args.checkpoint_dir) / 'optuna_study.db'}")
    print("\n💡 Tips:")
    print("  • Each worker pulls trials from shared Optuna database")
    print("  • Workers run independently and may finish at different times")
    print("  • Press Ctrl+C to stop all workers gracefully")
    print("  • If interrupted, resume with the same command")
    print("\n⏳ Waiting for workers to complete...\n")

    # Wait for all workers to complete
    completed = []
    try:
        while len(completed) < num_gpus:
            for i, proc in enumerate(processes):
                if i in completed:
                    continue

                retcode = proc.poll()
                if retcode is not None:
                    completed.append(i)
                    if retcode == 0:
                        print(f"  ✅ GPU {i} worker completed successfully")
                    else:
                        print(f"  ⚠️  GPU {i} worker exited with code {retcode}")

            time.sleep(5)

    except KeyboardInterrupt:
        signal_handler(None, None)

    print("\n" + "="*70)
    print("ALL WORKERS COMPLETED")
    print("="*70)

    failures = [i for i, proc in enumerate(processes) if proc.returncode != 0]
    if failures:
        print(f"\n⚠️  Some workers failed: GPUs {failures}")
        print("   Check logs for details")
    else:
        print("\n✅ All workers completed successfully!")

    print(f"\n📁 Results saved to: {args.checkpoint_dir}")
    print(f"   • Summary: {Path(args.checkpoint_dir) / 'summary'}")
    print(f"   • Best model: {Path(args.checkpoint_dir) / 'best_overall'}")
    print("="*70 + "\n")


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    """Main entry point for hyperparameter search with automatic GPU detection."""

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Arabic Writer Identification - Hyperparameter Search (Auto Multi-GPU)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
GPU Modes (Automatic):
  • 1 GPU: Sequential trials on single GPU
  • 2+ GPUs: Parallel trials (one trial per GPU) for 4x speedup

Example:
  python run_hyperparameter_search.py \\
    --data_root /path/to/data \\
    --checkpoint_dir /path/to/checkpoints \\
    --n_trials 24 \\
    --use_all_writers
        """
    )
    parser.add_argument('--data_root', type=str, required=True,
                   help='Path to Mirath_extracted_lines directory')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                   help='Directory to save checkpoints and results')
    parser.add_argument('--n_trials', type=int, default=12,
                   help='Total number of trials to run (default: 12)')
    parser.add_argument('--use_all_writers', action='store_true',
                   help='Use all writers')
    parser.add_argument('--num_writers_subset', type=int, default=7,
                   help='Number of writers if not using all (default: 7)')
    parser.add_argument('--single_gpu_mode', action='store_true',
                   help='Force single GPU mode (internal flag for parallel workers)')
    parser.add_argument('--quick_test', action='store_true',
                   help='Quick test mode: 3 writers, 50 lines/writer, 32 batch size, 5 epochs, 2 trials')

    args = parser.parse_args()

    # ========== QUICK TEST MODE ==========
    # Override parameters for fast testing
    if args.quick_test:
        print("\n" + "🚀"*35)
        print("QUICK TEST MODE ENABLED")
        print("🚀"*35)
        print("\n⚡ Quick Test Parameters:")
        print("  • Writers: 3 (instead of all 21)")
        print("  • Lines per writer: 50 (instead of 300)")
        print("  • Batch size: 32 (instead of 128)")
        print("  • Epochs: 5 (instead of 30)")
        print("  • Trials: 2 (instead of 12)")
        print("  • Max pages per writer (val): 3 (instead of 20)")
        print("\n⏱️  Estimated time: ~5-10 minutes per trial")
        print("🎯 Purpose: Verify code works before full run\n")
        print("="*70 + "\n")

        # Override arguments
        args.n_trials = 2
        args.num_writers_subset = 3
        args.use_all_writers = False  # Force subset mode

    # Detect number of GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if num_gpus == 0:
        print("❌ No GPUs available. This script requires CUDA-capable GPUs.")
        sys.exit(1)

    # Decide mode: parallel or single GPU
    if args.single_gpu_mode or num_gpus == 1:
        # Single GPU mode
        print(f"\n{'='*70}")
        print("SINGLE GPU MODE")
        print(f"{'='*70}")
        if args.single_gpu_mode:
            print("Running as worker in parallel mode")
        else:
            print(f"Detected: {num_gpus} GPU")
        print(f"{'='*70}\n")

        # Set seed
        set_seed(42)

        # Create base config
        config_base = BaseConfig()
        config_base.DATA_ROOT = args.data_root
        config_base.CHECKPOINT_DIR = args.checkpoint_dir
        config_base.USE_ALL_WRITERS = args.use_all_writers
        config_base.NUM_WRITERS_SUBSET = args.num_writers_subset
        config_base.QUICK_TEST = args.quick_test

        # Override parameters for quick test
        if args.quick_test:
            config_base.EPOCHS = 5  # Fast iterations
            config_base.MAX_PAGES_PER_WRITER_VAL = 3  # Less validation data
            config_base.EARLY_STOPPING_PATIENCE = 3  # Stop faster

        # Run hyperparameter search
        run_hyperparameter_search(config_base, args.n_trials)

    else:
        # Multi-GPU parallel mode
        run_parallel_search(args, num_gpus)


if __name__ == '__main__':
    main()


print("✅ Section 8: Main execution and summary analysis complete")
print("\n" + "="*70)
print("AUTO MULTI-GPU HYPERPARAMETER SEARCH SCRIPT READY")
print("="*70)
print("\nFeatures:")
print("  • 1 GPU: Automatic sequential mode")
print("  • 2+ GPUs: Automatic parallel mode (one trial per GPU)")
print("  • Fully resumable with SQLite database")
print("  • Graceful shutdown with Ctrl+C")
print("\nUsage:")
print("  python run_hyperparameter_search.py \\")
print("    --data_root /path/to/Mirath_extracted_lines \\")
print("    --checkpoint_dir /path/to/checkpoints \\")
print("    --n_trials 24 \\")
print("    --use_all_writers")
print("\n" + "="*70)

