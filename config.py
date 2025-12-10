"""
Configuration for INR network architecture and training.
"""

from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class NetworkConfig:
    """Network architecture configuration."""
    num_frequencies: int = 10
    hidden_dim: int = 256
    num_hidden_layers: int = 4
    include_input: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    num_epochs: int = 2000
    batch_size: int = 4096
    learning_rate: float = 1e-3
    device: Optional[str] = None  # None = auto-detect

    def get_device(self) -> torch.device:
        """Get the training device (auto-detect if not specified)."""
        if self.device is not None:
            return torch.device(self.device)

        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')


@dataclass
class Config:
    """Complete configuration for INR training."""
    network: NetworkConfig = None
    training: TrainingConfig = None

    def __post_init__(self):
        if self.network is None:
            self.network = NetworkConfig()
        if self.training is None:
            self.training = TrainingConfig()
