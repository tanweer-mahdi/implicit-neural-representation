"""
Training loop for INR.

Implements the core training logic to fit a network to an image.
"""

import torch
import torch.nn as nn
from typing import List


def train_inr(
    model: nn.Module,
    coords: torch.Tensor,
    rgb: torch.Tensor,
    num_epochs: int = 2000,
    batch_size: int = 4096,
    learning_rate: float = 1e-3,
    print_every: int = 100
) -> List[float]:
    """
    Train INR model on coordinate-RGB pairs.

    Args:
        model: INRNetwork to train
        coords: (N, 2) coordinates in [-1, 1]
        rgb: (N, 3) RGB values in [0, 1]
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for Adam optimizer
        print_every: Print progress every N epochs

    Returns:
        List of loss values (one per epoch)
    """
    # Setup
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    losses = []

    num_pixels = len(coords)
    model.train()

    # Training loop
    for epoch in range(num_epochs):
        # Sample random batch
        indices = torch.randint(0, num_pixels, (batch_size,), device=coords.device)
        batch_coords = coords[indices]
        batch_rgb = rgb[indices]

        # Forward pass
        predicted_rgb = model(batch_coords)
        loss = criterion(predicted_rgb, batch_rgb)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record loss
        losses.append(loss.item())

        # Print progress
        if (epoch + 1) % print_every == 0:
            psnr = -10 * torch.log10(loss)
            print(f"Epoch {epoch+1:4d} | Loss: {loss.item():.6f} | PSNR: {psnr.item():.2f} dB")

    return losses


def calculate_psnr(mse: float) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio from MSE.

    Args:
        mse: Mean squared error

    Returns:
        PSNR in dB
    """
    return -10 * torch.log10(torch.tensor(mse)).item()
