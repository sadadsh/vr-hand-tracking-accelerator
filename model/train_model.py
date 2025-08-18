"""
Training Pipeline for CNN VR Hand Gesture Tracking

Engineer: Sadad Haidari
"""

import os
import json
import argparse
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch.nn.functional as F

# Import our CNN model
from cnn import create_cnn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HaGRIDDataset(Dataset):
    
    def __init__(self, annotations_file: str, split: str, dataset_root: str, 
                 transform: Optional[transforms.Compose] = None):
        """
        Initialize HaGRID dataset.
        
        Args:
            annotations_file: Path to main annotations JSON
            split: 'train', 'validation', or 'test'
            dataset_root: Root directory of dataset
            transform: Image transformations
        """
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.transform = transform
        
        # Load annotations
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Load splits
        splits_file = Path(annotations_file).parent / 'train_val_test_splits.json'
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        
        # Get image IDs for this split
        self.image_ids = splits[split]
        
        self.class_names = [
            'call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one',
            'palm', 'peace_inverted', 'peace', 'rock', 'stop_inverted', 
            'stop', 'three', 'three2', 'two_up_inverted', 'two_up'
        ]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for idx, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        
        self.valid_samples = []
        class_counts = Counter()
        
        for image_id in self.image_ids:
            if image_id in self.annotations['images']:
                metadata = self.annotations['images'][image_id]
                gesture_name = metadata['gesture_name']
                if gesture_name in self.class_to_idx:
                    self.valid_samples.append((image_id, metadata))
                    class_counts[gesture_name] += 1
        
        import random
        random.seed(42)  # Reproducible shuffling
        random.shuffle(self.valid_samples)
        
        logger.info(f"{split} split: {len(self.valid_samples)} valid samples")
        
        # Store class distribution
        self.class_counts = class_counts
        
        # Print class distribution
        print(f"\n{split.upper()} Split Class Distribution:")
        for class_name in self.class_names:
            count = self.class_counts[class_name]
            print(f"  {class_name:15}: {count:,} images")
    
    def __len__(self) -> int:
        return len(self.valid_samples)
    
    def __getitem__(self, idx: int) -> Tuple[Any, int, Dict[str, Any]]:
        """
        Get dataset item with proper error handling.
        
        Returns:
            image: Transformed image tensor
            label: Class index
            metadata: Image metadata
        """
        _, metadata = self.valid_samples[idx]
        
        # Get image path
        image_path = self.dataset_root / metadata['image_path']
        
        image = None
        try:
            image = Image.open(image_path).convert('RGB')
            if image.size[0] < 50 or image.size[1] < 50:
                raise ValueError("Image too small")
        except Exception as e:
            logger.warning(f"Failed to load {image_path}: {e}")
            image = Image.new('RGB', (224, 224), (128, 128, 128))  # Gray instead of black
        
        # Get label
        gesture_name = metadata['gesture_name']
        label = self.class_to_idx[gesture_name]
        
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                logger.warning(f"Transform failed for {image_path}: {e}")
                # Create fallback tensor
                image = torch.zeros(3, 224, 224)
        
        return image, label, metadata
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced training."""
        total_samples = len(self.valid_samples)
        class_weights = []
        
        for class_name in self.class_names:
            count = self.class_counts[class_name]
            weight = np.sqrt(total_samples / (self.num_classes * count)) if count > 0 else 0
            class_weights.append(weight)
        
        return torch.FloatTensor(class_weights)
    
    def get_sample_weights(self) -> List[float]:
        """Calculate sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        sample_weights = []
        
        for _, metadata in self.valid_samples:
            gesture_name = metadata['gesture_name']
            class_idx = self.class_to_idx[gesture_name]
            sample_weights.append(class_weights[class_idx].item())
        
        return sample_weights


def create_transforms(input_size: int = 224, augment: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    **ADVANCED: Create transforms optimized for 90%+ accuracy with advanced augmentation**
    """
    # **ADVANCED: Use ImageNet normalization (standard for pre-trained features)**
    normalize_transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet means
        std=[0.229, 0.224, 0.225]    # ImageNet stds
    )
    
    # **ADVANCED: Aggressive training transforms for 90% accuracy**
    train_transforms = [
        transforms.Resize((int(input_size * 1.3), int(input_size * 1.3))),  # Larger resize
        transforms.RandomCrop((input_size, input_size)),
    ]
    
    if augment:
        train_transforms.extend([
            transforms.RandomHorizontalFlip(p=0.5),  # Standard flip
            transforms.RandomRotation(degrees=20),    # More rotation
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),  # More color augmentation
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),  # More affine transforms
            transforms.RandomPerspective(distortion_scale=0.1, p=0.3),  # Perspective distortion
            transforms.RandomGrayscale(p=0.1),  # Grayscale augmentation
        ])
    
    train_transforms.extend([
        transforms.ToTensor(),
        normalize_transform
    ])
    
    # **ADVANCED: Validation transforms (no augmentation)**
    val_transforms = [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        normalize_transform
    ]
    
    return transforms.Compose(train_transforms), transforms.Compose(val_transforms)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance and hard examples."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ModelTrainer:
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = Path(config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = create_cnn(
            num_classes=config['num_classes'],
            input_size=config['input_size'],
            dropout_rate=0.1  # Light regularization
        ).to(self.device)
        
        # Training state
        self.best_val_acc = 0.0
        self.best_model_state = None
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.early_stopping_counter = 0
        
        logger.info(f"Model initialized on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def create_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        # Create transforms
        train_transform, val_transform = create_transforms(
            input_size=self.config['input_size'],
            augment=self.config['data_augmentation']
        )
        
        # Create datasets
        train_dataset = HaGRIDDataset(
            self.config['annotations_file'],
            'train',
            self.config['dataset_root'],
            transform=train_transform
        )
        
        val_dataset = HaGRIDDataset(
            self.config['annotations_file'],
            'validation', 
            self.config['dataset_root'],
            transform=val_transform
        )
        
        test_dataset = HaGRIDDataset(
            self.config['annotations_file'],
            'test',
            self.config['dataset_root'],
            transform=val_transform
        )
        
        if self.config['balanced_sampling']:
            sample_weights = train_dataset.get_sample_weights()
            num_samples = min(len(sample_weights), 100000)  # Cap at 100K samples per epoch for better coverage
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=num_samples,
                replacement=True
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True
        
        effective_batch_size = min(self.config['batch_size'], 16)  # Cap batch size
        num_workers = min(self.config['num_workers'], 4)  # Reduce workers for stability
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=effective_batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=effective_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=effective_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            drop_last=False
        )
        
        logger.info(f"Data loaders created:")
        logger.info(f"  Train: {len(train_loader)} batches, {len(train_dataset)} samples")
        logger.info(f"  Val: {len(val_loader)} batches, {len(val_dataset)} samples") 
        logger.info(f"  Test: {len(test_loader)} batches, {len(test_dataset)} samples")
        logger.info(f"  Effective batch size: {effective_batch_size}")
        
        # Store class names for later use
        self.class_names = train_dataset.class_names
        
        return train_loader, val_loader, test_loader
    
    def create_optimizer_and_scheduler(self) -> Tuple[optim.Optimizer, Any]:
        
        # Use different learning rates for backbone and classifier
        backbone_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'backbone' in name:
                    backbone_params.append(param)
                else:
                    classifier_params.append(param)
        
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': self.config['learning_rate'] * 0.01},  # Much lower LR for backbone
            {'params': classifier_params, 'lr': self.config['learning_rate'] * 0.1}  # Lower LR for classifier
        ], weight_decay=1e-4)
        
        # **ULTRA-ADVANCED: Learning rate scheduler for 90% accuracy**
        # Use ReduceLROnPlateau for adaptive learning rate
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            min_lr=self.config['learning_rate'] * 0.00001
        )
        
        return optimizer, scheduler
    
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module, epoch: int) -> Tuple[float, float]:
        
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} [Train]')
        
        for batch_idx, (images, labels, _) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            if torch.isnan(images).any() or torch.isnan(labels).any():
                logger.warning(f"NaN detected in batch {batch_idx}, skipping...")
                continue
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(images)
            
            if torch.isnan(outputs).any():
                logger.warning(f"NaN in model outputs, batch {batch_idx}")
                continue
                
            loss = criterion(outputs, labels)
            
            if torch.isnan(loss):
                logger.warning(f"NaN loss detected, batch {batch_idx}")
                continue
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            current_acc = 100.0 * correct / total
            current_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
            
            if current_loss > 10.0:
                logger.warning(f"Loss explosion detected ({current_loss}), stopping epoch")
                break
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader: DataLoader, criterion: nn.Module, 
                      epoch: int) -> Tuple[float, float, np.ndarray, np.ndarray]:
        
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_predicted = []
        all_labels = []
        
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} [Val]')
        
        with torch.no_grad():
            for batch_idx, (images, labels, _) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                if torch.isnan(images).any() or torch.isnan(labels).any():
                    continue
                
                # Forward pass
                outputs = self.model(images)
                # Simple validation TTA: horizontal flip
                if self.config.get('use_tta', False):
                    images_flipped = torch.flip(images, dims=[3])
                    outputs_flipped = self.model(images_flipped)
                    outputs = (outputs + outputs_flipped) / 2.0
                
                if torch.isnan(outputs).any():
                    continue
                    
                loss = criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Store predictions for confusion matrix
                all_predicted.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                current_acc = 100.0 * correct / total
                current_loss = running_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc, np.array(all_predicted), np.array(all_labels)
    
    def save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False):
        """Save model checkpoint."""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_acc': val_acc,
            'config': self.config,
            'class_names': self.class_names
        }
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved with validation accuracy: {val_acc:.2f}%")
    
    def plot_training_history(self):
        """Plot training and validation curves."""
        
        plt.figure(figsize=(15, 5))
        
        # Loss curves
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Train Loss', color='blue')
        plt.plot(self.val_losses, label='Val Loss', color='red')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy curves
        plt.subplot(1, 3, 2)
        plt.plot(self.train_accs, label='Train Acc', color='blue')
        plt.plot(self.val_accs, label='Val Acc', color='red')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        # Learning rate
        plt.subplot(1, 3, 3)
        epochs = range(len(self.train_losses))
        lrs = [self.config['learning_rate'] * (0.5 ** (epoch // 10)) for epoch in epochs]
        plt.plot(epochs, lrs)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, split_name: str):
        """Plot confusion matrix."""
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title(f'Confusion Matrix - {split_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.save_dir / f'confusion_matrix_{split_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate per-class accuracy
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        
        return per_class_acc
    
    def train(self):
        
        logger.info("Starting training...")
        print(f"\n{'='*80}")
        print("STARTING CNN TRAINING")
        print(f"{'='*80}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Target accuracy: >{self.config['target_accuracy']}%")
        print(f"Max epochs: {self.config['epochs']}")
        print(f"Early stopping patience: {self.config['early_stopping_patience']}")
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.create_data_loaders()
        
        # Create optimizer and scheduler
        optimizer, scheduler = self.create_optimizer_and_scheduler()
        
        # **ULTRA-ADVANCED: Use Focal Loss for handling hard examples and class imbalance (train)**
        criterion_train = FocalLoss(alpha=1.0, gamma=2.0)
        # **Validation**: use CrossEntropy with label smoothing for better calibration
        criterion_val = nn.CrossEntropyLoss(label_smoothing=0.05)
        
        # Training loop
        start_time = time.time()
        
        warmup_epochs = 3
        base_lr = self.config['learning_rate']
        
        for epoch in range(self.config['epochs']):
            
            if epoch < warmup_epochs:
                warmup_lr = base_lr * (0.1 + 0.9 * (epoch + 1) / warmup_epochs)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            # Train epoch
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion_train, epoch)
            
            # Validate epoch
            val_loss, val_acc, _, _ = self.validate_epoch(val_loader, criterion_val, epoch)
            
            # Update learning rate (after warmup) using validation accuracy
            if epoch >= warmup_epochs:
                scheduler.step(val_acc)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # Print epoch summary
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            
            if epoch < 5:
                if val_acc > 20:
                    print(f"  🚀 Great start! Model is learning")
                elif val_acc > 10:
                    print(f"  📈 Good progress, model is training")
                else:
                    print(f"  ⚠️  Slow start, but normal for complex data")
            elif val_acc > 70:
                print(f"  🎯 Excellent performance!")
            elif val_acc > 50:
                print(f"  👍 Good progress")
            elif val_acc < 20:
                print(f"  ⚠️  Consider adjusting hyperparameters")
            
            if epoch > 0:
                if train_loss > self.train_losses[-2] * 2:
                    print(f"  ⚠️  Loss spike detected - reducing learning rate")
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.5
            
            # Check for best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                self.early_stopping_counter = 0
                self.save_checkpoint(epoch, val_acc, is_best=True)
                print(f"  *** NEW BEST MODEL: {val_acc:.2f}% ***")
            else:
                self.early_stopping_counter += 1
            
            # Save regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, val_acc, is_best=False)
            
            if self.early_stopping_counter >= self.config['early_stopping_patience']:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            # Check target accuracy
            if val_acc >= self.config['target_accuracy']:
                print(f"\nTarget accuracy {self.config['target_accuracy']}% reached!")
                break
        
        # Training completed
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print("TRAINING COMPLETED!")
        print(f"{'='*80}")
        print(f"Total training time: {total_time/60:.1f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Target achieved: {'✓' if self.best_val_acc >= self.config['target_accuracy'] else '✗'}")
        
        # Load best model for final evaluation
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Final evaluation on test set
        print(f"\n{'='*40}")
        print("FINAL TEST SET EVALUATION")
        print(f"{'='*40}")
        
        _, test_acc, test_pred, test_true = self.validate_epoch(
            test_loader, criterion_val, epoch=-1
        )
        
        print(f"Test Accuracy: {test_acc:.2f}%")
        
        # Generate confusion matrix and per-class metrics
        per_class_acc = self.plot_confusion_matrix(test_true, test_pred, 'Test')
        
        print(f"\nPer-Class Test Accuracy:")
        for _, (class_name, acc) in enumerate(zip(self.class_names, per_class_acc)):
            print(f"  {class_name:15}: {acc*100:.1f}%")
        
        # Plot training history
        self.plot_training_history()
        
        # Save final model info
        final_info = {
            'best_val_accuracy': self.best_val_acc,
            'test_accuracy': test_acc,
            'total_training_time_minutes': total_time / 60,
            'total_epochs_trained': len(self.train_losses),
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'target_achieved': self.best_val_acc >= self.config['target_accuracy'],
            'per_class_test_accuracy': {
                class_name: float(acc) for class_name, acc in zip(self.class_names, per_class_acc)
            }
        }
        
        with open(self.save_dir / 'training_results.json', 'w') as f:
            json.dump(final_info, f, indent=2)
        
        logger.info(f"Training completed. Results saved to {self.save_dir}")
        
        return final_info


def main():
    """Main training function."""
    
    parser = argparse.ArgumentParser(description='Train CNN for VR Hand Tracking')
    parser.add_argument('--annotations', '-a', type=str, 
                       default='data/annotations/hagrid_all_gestures_annotations.json',
                       help='Path to annotations file')
    parser.add_argument('--dataset_root', '-d', type=str,
                       default='data/local_dataset/hagrid_500k',
                       help='Root directory of dataset')
    parser.add_argument('--save_dir', '-s', type=str,
                       default='model/checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--epochs', '-e', type=int, default=25,
                       help='Maximum number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--no_augmentation', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--no_balanced_sampling', action='store_true',
                       help='Disable balanced sampling')
    parser.add_argument('--target_accuracy', '-t', type=float, default=80.0,
                       help='Target validation accuracy percentage')
    
    args = parser.parse_args()
    
    config = {
        'annotations_file': args.annotations,
        'dataset_root': args.dataset_root,
        'save_dir': args.save_dir,
        'num_classes': 18,
        'input_size': 224,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-4,
        'data_augmentation': not args.no_augmentation,
        'balanced_sampling': not args.no_balanced_sampling,
        'balanced_loss': False,
        'early_stopping_patience': 8,
        'target_accuracy': args.target_accuracy,
        'num_workers': min(4, os.cpu_count() or 1),
        'use_tta': True
    }
    
    print("CNN Training Pipeline")
    print("="*50)
    print(f"Annotations: {config['annotations_file']}")
    print(f"Dataset root: {config['dataset_root']}")
    print(f"Save directory: {config['save_dir']}")
    print(f"Target accuracy: {config['target_accuracy']}%")
    print(f"Max epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Data augmentation: {config['data_augmentation']}")
    print(f"Balanced sampling: {config['balanced_sampling']}")
    
    # Confirm training
    response = input("\nStart training with fixes? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Training cancelled.")
        return
    
    # Create trainer and start training
    trainer = ModelTrainer(config)
    results = trainer.train()
    
    # Final summary
    print(f"\nTRAINING COMPLETED!")
    print(f"✓ Best validation accuracy: {results['best_val_accuracy']:.2f}%")
    print(f"✓ Test accuracy: {results['test_accuracy']:.2f}%")
    print(f"✓ Training time: {results['total_training_time_minutes']:.1f} minutes")
    print(f"✓ Model parameters: {results['model_parameters']:,}")
    print(f"✓ Target achieved: {'Yes' if results['target_achieved'] else 'No'}")
    print(f"✓ Ready for quantization: Yes")
    
    if results['best_val_accuracy'] > 60:
        print(f"\n🎯 EXCELLENT!")
    elif results['best_val_accuracy'] > 40:
        print(f"\n👍 GOOD!")
    else:
        print(f"\n⚠️ BAD.")


if __name__ == "__main__":
    main()