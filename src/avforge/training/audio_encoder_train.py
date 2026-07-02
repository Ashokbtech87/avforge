"""
Audio encoder training script for Phase 1.
Trains a lightweight audio encoder on mel-spectrogram features.
Optimized for 8GB VRAM (RTX 4060).
"""

import logging
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import torch.nn.functional as F

from avforge.data.audio_dataset import create_audio_dataloaders
from avforge.encoders import AudioEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioEncoderTrainer:
    """Trainer for audio encoders with memory optimization for 8GB VRAM."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        output_dir: Path = Path('results/audio_encoder'),
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_steps: int = 500,
        max_grad_norm: float = 1.0,
        use_fp16: bool = True,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Audio encoder model
            device: torch.device
            output_dir: Directory to save checkpoints
            learning_rate: Initial learning rate
            weight_decay: Weight decay for optimizer
            warmup_steps: Number of warmup steps
            max_grad_norm: Maximum gradient norm for clipping
            use_fp16: Use mixed precision training
        """
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.use_fp16 = use_fp16
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,  # Will be updated
            eta_min=1e-6,
        )
        
        # Mixed precision
        self.scaler = GradScaler('cuda') if use_fp16 else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
    
    def _get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def _warmup_lr_schedule(self, step: int) -> float:
        """Learning rate warmup schedule."""
        if step < self.warmup_steps:
            return self.learning_rate * (step / self.warmup_steps)
        else:
            return self.learning_rate
    
    def _update_lr(self, step: int):
        """Update learning rate with warmup."""
        if step < self.warmup_steps:
            lr = self._warmup_lr_schedule(step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
    
    def _compute_contrastive_loss(
        self,
        mel_embeddings: torch.Tensor,
        mfcc_embeddings: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """
        Compute contrastive loss between mel-spec and MFCC embeddings.
        
        Args:
            mel_embeddings: (batch, embedding_dim)
            mfcc_embeddings: (batch, embedding_dim)
            temperature: Temperature for contrastive loss
        
        Returns:
            Scalar loss
        """
        # Normalize embeddings
        mel_embeddings = F.normalize(mel_embeddings, dim=-1)
        mfcc_embeddings = F.normalize(mfcc_embeddings, dim=-1)
        
        # Compute similarity matrix
        batch_size = mel_embeddings.shape[0]
        
        # Positive pairs: same sample across modalities
        similarity_matrix = torch.matmul(
            mel_embeddings,
            mfcc_embeddings.T
        ) / temperature
        
        # Labels: diagonal elements should be high
        labels = torch.arange(batch_size, device=mel_embeddings.device)
        
        # Cross-entropy loss
        loss_i = F.cross_entropy(similarity_matrix, labels)
        loss_j = F.cross_entropy(similarity_matrix.T, labels)
        
        return (loss_i + loss_j) / 2.0
    
    def train_step(self, batch: Dict) -> Tuple[float, float]:
        """
        Single training step.
        
        Returns:
            Tuple of (loss, learning_rate)
        """
        self.model.train()
        
        mel_spec = batch['mel_spec'].to(self.device)
        mel_mask = batch['mel_mask'].to(self.device)
        
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision
        # Use mel-spec for both views with different dropout for contrastive learning
        if self.use_fp16:
            with autocast('cuda'):
                # View 1: standard mel-spec
                mel_embeddings = self.model(mel_spec, attention_mask=mel_mask)
                # View 2: same mel-spec with model in train mode (different dropout)
                mel_embeddings_2 = self.model(mel_spec, attention_mask=mel_mask)
                loss = self._compute_contrastive_loss(mel_embeddings, mel_embeddings_2)
        else:
            mel_embeddings = self.model(mel_spec, attention_mask=mel_mask)
            mel_embeddings_2 = self.model(mel_spec, attention_mask=mel_mask)
            loss = self._compute_contrastive_loss(mel_embeddings, mel_embeddings_2)
        
        # Backward pass
        if self.use_fp16:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        # Optimizer step
        if self.use_fp16:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Learning rate update
        self._update_lr(self.global_step)
        self.global_step += 1
        
        return loss.item(), self._get_lr()
    
    def eval_step(self, loader: DataLoader) -> float:
        """
        Evaluation step.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in loader:
                mel_spec = batch['mel_spec'].to(self.device)
                mel_mask = batch['mel_mask'].to(self.device)
                
                if self.use_fp16:
                    with autocast('cuda'):
                        mel_embeddings = self.model(mel_spec, attention_mask=mel_mask)
                        mel_embeddings_2 = self.model(mel_spec, attention_mask=mel_mask)
                        loss = self._compute_contrastive_loss(mel_embeddings, mel_embeddings_2)
                else:
                    mel_embeddings = self.model(mel_spec, attention_mask=mel_mask)
                    mel_embeddings_2 = self.model(mel_spec, attention_mask=mel_mask)
                    loss = self._compute_contrastive_loss(mel_embeddings, mel_embeddings_2)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def save_checkpoint(self, name: str = 'latest'):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        save_path = self.output_dir / f'checkpoint_{name}.pt'
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint: {save_path}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        log_interval: int = 10,
        eval_interval: int = 1,
    ):
        """
        Full training loop.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_epochs: Number of training epochs
            log_interval: Log interval in steps
            eval_interval: Evaluation interval in epochs
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Using FP16: {self.use_fp16}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Update scheduler
        self.scheduler.T_max = num_epochs
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            num_batches = 0
            
            # Training loop
            for step, batch in enumerate(train_loader):
                loss, lr = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1
                
                if (step + 1) % log_interval == 0:
                    avg_loss = epoch_loss / num_batches
                    logger.info(
                        f"Epoch {epoch+1}/{num_epochs} | "
                        f"Step {step+1} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr:.2e}"
                    )
            
            # Average epoch loss
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            self.train_losses.append(avg_epoch_loss)
            
            # Validation
            if (epoch + 1) % eval_interval == 0:
                val_loss = self.eval_step(val_loader)
                self.val_losses.append(val_loss)
                
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Train Loss: {avg_epoch_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f}"
                )
                
                # Save best checkpoint
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint('best')
            
            # Save latest checkpoint
            self.save_checkpoint('latest')
            self.scheduler.step()
        
        logger.info("Training complete!")
        
        # Save metrics
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_loss': self.best_loss,
        }
        metrics_path = self.output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics: {metrics_path}")


def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train audio encoder on GokuScraper Seedance 2.0 videos'
    )
    parser.add_argument(
        '--feature-dir',
        default='data/seedance_prompts/audio_features',
        help='Directory with precomputed audio features'
    )
    parser.add_argument(
        '--metadata-path',
        default='data/seedance_prompts/raw_hf/metadata.jsonl',
        help='Path to metadata JSONL file'
    )
    parser.add_argument(
        '--output-dir',
        default='results/audio_encoder',
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=2,
        help='Batch size (default: 2 for 8GB VRAM)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=500,
        help='Number of warmup steps'
    )
    parser.add_argument(
        '--no-fp16',
        action='store_true',
        help='Disable mixed precision training'
    )
    parser.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training'
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create model
    model = AudioEncoder(
        input_dim=128,  # mel-spectrogram
        hidden_dim=256,
        embedding_dim=128,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
    )
    logger.info(f"Created AudioEncoder with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_audio_dataloaders(
        feature_dir=Path(args.feature_dir),
        metadata_path=Path(args.metadata_path),
        batch_size=args.batch_size,
        num_workers=0,  # Set to 0 on Windows
    )
    logger.info(f"Train: {len(train_loader)} batches | Val: {len(val_loader)} batches | Test: {len(test_loader)} batches")
    
    # Create trainer
    trainer = AudioEncoderTrainer(
        model=model,
        device=device,
        output_dir=Path(args.output_dir),
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        use_fp16=not args.no_fp16,
    )
    
    # Train
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        log_interval=10,
        eval_interval=1,
    )


if __name__ == '__main__':
    main()
