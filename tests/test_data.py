"""
Tests for data preparation module.
Run with: uv run python tests/test_data.py
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.inr import load_image, create_coordinate_grid, prepare_training_data


def test_load_image():
    """Test loading and preprocessing image."""
    image_path = Path(__file__).parent.parent / "test_image.jpg"
    image = load_image(image_path, max_size=256)

    assert image.shape[2] == 3  # RGB channels
    assert image.min() >= 0.0 and image.max() <= 1.0  # Values in [0, 1]
    assert image.dtype == torch.float32
    print(f"✓ Load image: shape {image.shape}, range [{image.min():.3f}, {image.max():.3f}]")


def test_coordinate_grid():
    """Test coordinate grid creation."""
    device = torch.device('cpu')
    coords = create_coordinate_grid(256, 256, device)

    assert coords.shape == (256 * 256, 2)
    assert coords.min() >= -1.0 and coords.max() <= 1.0
    assert coords.device.type == 'cpu'
    print(f"✓ Coordinate grid: shape {coords.shape}, range [{coords.min():.3f}, {coords.max():.3f}]")


def test_prepare_training_data():
    """Test complete data preparation pipeline."""
    image_path = Path(__file__).parent.parent / "test_image.jpg"
    device = torch.device('cpu')

    coords, rgb, (H, W) = prepare_training_data(image_path, device, max_size=256)

    assert coords.shape == (H * W, 2)
    assert rgb.shape == (H * W, 3)
    assert coords.device == rgb.device == device
    print(f"✓ Training data: {H}×{W} image → {coords.shape[0]:,} coordinate-RGB pairs")


def test_device_placement():
    """Test that data is placed on correct device."""
    image_path = Path(__file__).parent.parent / "test_image.jpg"

    # CPU
    coords_cpu, rgb_cpu, _ = prepare_training_data(image_path, torch.device('cpu'))
    assert coords_cpu.device.type == 'cpu'
    assert rgb_cpu.device.type == 'cpu'
    print(f"✓ CPU placement: working")

    # MPS if available
    if torch.backends.mps.is_available():
        coords_mps, rgb_mps, _ = prepare_training_data(image_path, torch.device('mps'))
        assert coords_mps.device.type == 'mps'
        assert rgb_mps.device.type == 'mps'
        print(f"✓ MPS placement: working")


def test_coordinate_range():
    """Test that coordinates are properly normalized to [-1, 1]."""
    device = torch.device('cpu')

    for H, W in [(64, 64), (128, 256), (256, 128)]:
        coords = create_coordinate_grid(H, W, device)
        assert torch.isclose(coords.min(), torch.tensor(-1.0), atol=0.01)
        assert torch.isclose(coords.max(), torch.tensor(1.0), atol=0.01)

    print(f"✓ Coordinate normalization: tested multiple sizes")


if __name__ == "__main__":
    print("=" * 70)
    print("Data Preparation Tests")
    print("=" * 70)

    tests = [
        test_load_image,
        test_coordinate_grid,
        test_prepare_training_data,
        test_device_placement,
        test_coordinate_range,
    ]

    passed = 0
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} failed: {e}")

    print("=" * 70)
    print(f"Result: {passed}/{len(tests)} tests passed")
    print("=" * 70)
