"""
INR Network - MLP that maps encoded coordinates to RGB colors.
"""

import torch
import torch.nn as nn
from typing import List

from .positional_encoding import PositionalEncoding


class INRNetwork(nn.Module):
    """
    MLP network for Implicit Neural Representation.

    Architecture:
        Coordinates (2D) → Positional Encoding → MLP → RGB (3D)
    """

    def __init__(
        self,
        num_frequencies: int = 10,
        hidden_dim: int = 256,
        num_hidden_layers: int = 4,
        include_input: bool = True,
    ):
        super().__init__()

        # Positional encoder
        self.encoder = PositionalEncoding(
            num_frequencies=num_frequencies,
            include_input=include_input
        )

        # Calculate input dimension
        input_dim = self.encoder._calculate_output_dim(input_dim=2)

        # Build MLP
        layers: List[nn.Module] = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))

        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        # Output layer
        layers.append(nn.Linear(hidden_dim, 3))
        layers.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: (batch, 2) in [-1, 1]
        returns: (batch, 3) in [0, 1]
        """
        encoded = self.encoder(coords)
        rgb = self.mlp(encoded)
        return rgb

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_inr_network(config, device: torch.device) -> INRNetwork:
    """
    Create network from config.

    Args:
        config: NetworkConfig with architecture parameters
        device: Device to place network on
    """
    model = INRNetwork(
        num_frequencies=config.num_frequencies,
        hidden_dim=config.hidden_dim,
        num_hidden_layers=config.num_hidden_layers,
        include_input=config.include_input
    )
    return model.to(device)
