"""
Data preparation for INR training.

Loads images and converts them to coordinate-RGB pairs for training.
"""

import torch
from PIL import Image
from pathlib import Path
from typing import Tuple, Union


def load_image(image_path: Union[str, Path], max_size: int = 256) -> torch.Tensor:
    """
    Load image and convert to tensor.

    Args:
        image_path: Path to image file
        max_size: Maximum dimension (will resize if larger)

    Returns:
        Image tensor of shape (H, W, 3) with values in [0, 1]
    """
    image = Image.open(image_path).convert('RGB')

    # Resize if needed (maintain aspect ratio)
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.LANCZOS)

    # Convert to tensor [0, 1]
    image_array = torch.tensor(list(image.getdata()), dtype=torch.float32) / 255.0
    image_tensor = image_array.reshape(image.size[1], image.size[0], 3)

    return image_tensor


def create_coordinate_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    """
    Create normalized coordinate grid.

    Args:
        height: Image height
        width: Image width
        device: Device to place tensor on

    Returns:
        Coordinates of shape (H*W, 2) with values in [-1, 1]
    """
    # Create coordinate grids
    y = torch.linspace(-1, 1, height, device=device)
    x = torch.linspace(-1, 1, width, device=device)

    # Create meshgrid
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    # Stack and reshape to (H*W, 2)
    coords = torch.stack([xx, yy], dim=-1)
    coords = coords.reshape(-1, 2)

    return coords


def prepare_training_data(
    image_path: Union[str, Path],
    device: torch.device,
    max_size: int = 256
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """
    Prepare image for training.

    Args:
        image_path: Path to image file
        device: Device to place tensors on
        max_size: Maximum image dimension

    Returns:
        coords: (N, 2) coordinates in [-1, 1]
        rgb: (N, 3) RGB values in [0, 1]
        image_shape: (H, W) original image dimensions
    """
    # Load image
    image = load_image(image_path, max_size)
    H, W = image.shape[:2]

    # Create coordinate grid
    coords = create_coordinate_grid(H, W, device)

    # Flatten RGB values
    rgb = image.reshape(-1, 3).to(device)

    return coords, rgb, (H, W)
