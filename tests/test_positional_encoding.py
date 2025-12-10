"""
Tests for Positional Encoding module.
Run with: uv run python tests/test_positional_encoding.py
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.inr import PositionalEncoding, positional_encode, get_encoding_dim


def test_basic_encoding():
    """Test basic positional encoding functionality."""
    encoder = PositionalEncoding(num_frequencies=10, include_input=True)
    coords = torch.tensor([[0.5, -0.3], [0.0, 0.0]])
    encoded = encoder(coords)

    expected_dim = 42  # 2 + 2*2*10
    assert encoded.shape == (2, expected_dim)
    print(f"✓ Basic encoding: {coords.shape} -> {encoded.shape}")


def test_batch_processing():
    """Test encoding with different batch sizes."""
    encoder = PositionalEncoding(num_frequencies=10)

    for batch_size in [1, 16, 256]:
        coords = torch.rand(batch_size, 2) * 2 - 1
        encoded = encoder(coords)
        assert encoded.shape == (batch_size, 42)

    print(f"✓ Batch processing: tested sizes 1, 16, 256")


def test_gradient_flow():
    """Test that gradients flow correctly through encoding."""
    encoder = PositionalEncoding(num_frequencies=10)
    coords = torch.rand(10, 2) * 2 - 1
    coords.requires_grad = True

    encoded = encoder(coords)
    loss = encoded.sum()
    loss.backward()

    assert coords.grad is not None
    assert not torch.isnan(coords.grad).any()
    print(f"✓ Gradient flow: working correctly")


def test_device_compatibility():
    """Test encoding on different devices."""
    # CPU
    encoder = PositionalEncoding(num_frequencies=10)
    coords = torch.rand(10, 2)
    encoded = encoder(coords)
    assert encoded.device.type == 'cpu'
    print(f"✓ CPU: working")

    # MPS if available
    if torch.backends.mps.is_available():
        encoder_mps = encoder.to('mps')
        coords_mps = coords.to('mps')
        encoded_mps = encoder_mps(coords_mps)
        assert encoded_mps.device.type == 'mps'
        print(f"✓ MPS: working")


def test_utility_functions():
    """Test helper functions."""
    # Functional interface
    coords = torch.rand(5, 2) * 2 - 1
    encoded = positional_encode(coords, num_frequencies=10)
    assert encoded.shape == (5, 42)

    # Dimension calculator
    dim = get_encoding_dim(input_dim=2, num_frequencies=10, include_input=True)
    assert dim == 42

    print(f"✓ Utility functions: working")


if __name__ == "__main__":
    print("=" * 70)
    print("Positional Encoding Tests")
    print("=" * 70)

    tests = [
        test_basic_encoding,
        test_batch_processing,
        test_gradient_flow,
        test_device_compatibility,
        test_utility_functions,
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
