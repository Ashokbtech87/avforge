"""
Audio dataset for training audio encoders.
Handles feature loading, transcription labels, and batching.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class AudioDataset(Dataset):
    """
    PyTorch Dataset for audio feature training.
    
    Expects:
    - Audio features as .npz files (mel-spectrogram, mfcc)
    - Metadata as JSONL file with prompts, duration, language
    """
    
    def __init__(
        self,
        feature_dir: Path,
        metadata_path: Path,
        split: str = 'train',
        max_duration: float = 15.0,
        min_duration: float = 4.0,
        sample_rate: int = 16000,
        n_mel_bins: int = 128,
    ):
        """
        Initialize audio dataset.
        
        Args:
            feature_dir: Directory containing precomputed .npz feature files
            metadata_path: Path to JSONL file with video metadata
            split: 'train', 'val', or 'test' split
            max_duration: Maximum video duration in seconds
            min_duration: Minimum video duration in seconds
            sample_rate: Audio sample rate (Hz)
            n_mel_bins: Number of mel-spectrogram bins
        """
        self.feature_dir = Path(feature_dir)
        self.metadata_path = Path(metadata_path)
        self.split = split
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.sample_rate = sample_rate
        self.n_mel_bins = n_mel_bins
        
        # Load metadata
        self.metadata = self._load_metadata()
        logger.info(f"Loaded {len(self.metadata)} samples for {split} split")
    
    def _load_metadata(self) -> List[Dict]:
        """Load and filter metadata from JSONL."""
        metadata = []
        
        if not self.metadata_path.exists():
            logger.warning(f"Metadata file not found: {self.metadata_path}")
            return metadata
        
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    entry = json.loads(line)
                    
                    # Filter by duration
                    duration = entry.get('spec', {}).get('duration', 10.0)
                    if duration < self.min_duration or duration > self.max_duration:
                        continue
                    
                    # Assign to split (deterministic based on ID hash)
                    video_id = entry.get('id', f'unknown_{i}')
                    split_idx = hash(video_id) % 10
                    
                    if self.split == 'train' and split_idx < 8:
                        metadata.append(entry)
                    elif self.split == 'val' and split_idx == 8:
                        metadata.append(entry)
                    elif self.split == 'test' and split_idx == 9:
                        metadata.append(entry)
                
                except json.JSONDecodeError:
                    continue
        
        return metadata
    
    def _load_features(self, idx: int) -> Optional[Dict[str, np.ndarray]]:
        """Load precomputed audio features for a sample."""
        try:
            entry = self.metadata[idx]
            video_id = entry.get('id', f'unknown_{idx}')
            
            # Try to load .npz feature file
            feature_path = self.feature_dir / f"{video_id}.npz"
            
            if not feature_path.exists():
                # Fallback: try to load from metadata
                logger.warning(f"Feature file not found: {feature_path}")
                return None
            
            # Load numpy archive
            data = np.load(str(feature_path))
            
            return {
                'mel_spec': data['mel_spec'].astype(np.float32),
                'mfcc': data['mfcc'].astype(np.float32),
            }
        except Exception as e:
            logger.error(f"Error loading features for sample {idx}: {e}")
            return None
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single audio sample with features and labels.
        
        Returns:
            Dict with keys:
            - 'mel_spec': (n_mel_bins, time_steps)
            - 'mfcc': (13, time_steps)
            - 'prompt': str
            - 'language': str
            - 'duration': float
        """
        entry = self.metadata[idx]
        
        # Load features
        features = self._load_features(idx)
        if features is None:
            # Return dummy features if file not found
            features = {
                'mel_spec': np.zeros((self.n_mel_bins, 100), dtype=np.float32),
                'mfcc': np.zeros((13, 100), dtype=np.float32),
            }
        
        # Extract metadata
        prompt = entry.get('en', {}).get('p', '')  # English prompt
        language = entry.get('i18n', {}).keys()
        duration = entry.get('spec', {}).get('duration', 10.0)
        
        return {
            'mel_spec': torch.from_numpy(features['mel_spec']),
            'mfcc': torch.from_numpy(features['mfcc']),
            'prompt': prompt,
            'language': list(language) if language else ['en'],
            'duration': duration,
            'video_id': entry.get('id', f'unknown_{idx}'),
        }


class AudioCollator:
    """Custom collate function for audio batches with variable lengths."""
    
    def __init__(self, pad_value: float = -100.0):
        """
        Initialize collator.
        
        Args:
            pad_value: Value to use for padding
        """
        self.pad_value = pad_value
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of audio samples.
        
        Pads to the maximum length in the batch.
        """
        # Extract sequences
        mel_specs = [item['mel_spec'] for item in batch]
        mfccs = [item['mfcc'] for item in batch]
        prompts = [item['prompt'] for item in batch]
        durations = torch.tensor([item['duration'] for item in batch])
        video_ids = [item['video_id'] for item in batch]
        
        # Find max lengths
        max_mel_steps = max(s.shape[1] for s in mel_specs)
        max_mfcc_steps = max(m.shape[1] for m in mfccs)
        
        # Pad mel-spectrograms
        mel_batch = []
        mel_mask = []
        for mel in mel_specs:
            pad_amount = max_mel_steps - mel.shape[1]
            if pad_amount > 0:
                mel = torch.nn.functional.pad(mel, (0, pad_amount), value=self.pad_value)
            mel_batch.append(mel)
            # Create attention mask (1 for real, 0 for padded)
            mask = torch.ones(max_mel_steps)
            mask[-pad_amount:] = 0 if pad_amount > 0 else 1
            mel_mask.append(mask)
        
        mel_batch = torch.stack(mel_batch)  # (batch, n_mel_bins, time)
        mel_mask = torch.stack(mel_mask)     # (batch, time)
        
        # Pad MFCCs
        mfcc_batch = []
        for mfcc in mfccs:
            pad_amount = max_mfcc_steps - mfcc.shape[1]
            if pad_amount > 0:
                mfcc = torch.nn.functional.pad(mfcc, (0, pad_amount), value=self.pad_value)
            mfcc_batch.append(mfcc)
        
        mfcc_batch = torch.stack(mfcc_batch)  # (batch, 13, time)
        
        return {
            'mel_spec': mel_batch,
            'mel_mask': mel_mask,
            'mfcc': mfcc_batch,
            'prompts': prompts,
            'durations': durations,
            'video_ids': video_ids,
        }


def create_audio_dataloaders(
    feature_dir: Path,
    metadata_path: Path,
    batch_size: int = 2,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders for audio training.
    
    Args:
        feature_dir: Directory with .npz feature files
        metadata_path: Path to JSONL metadata file
        batch_size: Batch size for training
        num_workers: Number of DataLoader workers
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    collator = AudioCollator(pad_value=-100.0)
    
    train_dataset = AudioDataset(
        feature_dir=feature_dir,
        metadata_path=metadata_path,
        split='train'
    )
    val_dataset = AudioDataset(
        feature_dir=feature_dir,
        metadata_path=metadata_path,
        split='val'
    )
    test_dataset = AudioDataset(
        feature_dir=feature_dir,
        metadata_path=metadata_path,
        split='test'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return train_loader, val_loader, test_loader
