"""
Tests for INRNetwork module.
Run with: uv run python tests/test_network.py
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.inr import INRNetwork, create_inr_network
from config import NetworkConfig


def test_basic_forward_pass():
    """Test basic forward pass through network."""
    model = INRNetwork()
    coords = torch.rand(10, 2) * 2 - 1
    rgb = model(coords)

    assert rgb.shape == (10, 3)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0
    print(f"✓ Forward pass: {coords.shape} -> {rgb.shape}")


def test_batch_sizes():
    """Test different batch sizes."""
    model = INRNetwork()

    for batch_size in [1, 16, 256, 1024]:
        coords = torch.rand(batch_size, 2) * 2 - 1
        rgb = model(coords)
        assert rgb.shape == (batch_size, 3)

    print(f"✓ Batch sizes: tested 1, 16, 256, 1024")


def test_gradient_flow():
    """Test that gradients flow through the network."""
    model = INRNetwork()
    coords = torch.rand(10, 2) * 2 - 1
    coords.requires_grad = True

    rgb = model(coords)
    loss = rgb.sum()
    loss.backward()

    assert coords.grad is not None
    for param in model.parameters():
        assert param.grad is not None

    print(f"✓ Gradient flow: working")


def test_device_compatibility():
    """Test network on different devices."""
    model = INRNetwork()
    coords = torch.rand(10, 2) * 2 - 1
    rgb = model(coords)
    assert rgb.device.type == 'cpu'
    print(f"✓ CPU: working")

    if torch.backends.mps.is_available():
        model_mps = model.to('mps')
        coords_mps = coords.to('mps')
        rgb_mps = model_mps(coords_mps)
        assert rgb_mps.device.type == 'mps'
        print(f"✓ MPS: working")


def test_config_based_creation():
    """Test creating network from config."""
    config = NetworkConfig(hidden_dim=128, num_hidden_layers=2)
    device = torch.device('cpu')
    model = create_inr_network(config, device)

    coords = torch.rand(10, 2) * 2 - 1
    rgb = model(coords)

    assert rgb.shape == (10, 3)
    print(f"✓ Config-based creation: working")
    print(f"  Parameters: {model.count_parameters():,}")


if __name__ == "__main__":
    print("=" * 70)
    print("INRNetwork Tests")
    print("=" * 70)

    tests = [
        test_basic_forward_pass,
        test_batch_sizes,
        test_gradient_flow,
        test_device_compatibility,
        test_config_based_creation,
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
