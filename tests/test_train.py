"""
Tests for training module.
Run with: uv run python tests/test_train.py
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.inr import INRNetwork, prepare_training_data, train_inr, calculate_psnr


def test_training_convergence():
    """Test that training loss decreases."""
    # Setup
    device = torch.device('cpu')  # Use CPU for faster test
    model = INRNetwork(num_frequencies=6, hidden_dim=128, num_hidden_layers=2).to(device)

    # Create small synthetic data
    coords = torch.rand(1000, 2, device=device) * 2 - 1
    rgb = torch.rand(1000, 3, device=device)

    # Train for a few epochs
    losses = train_inr(model, coords, rgb, num_epochs=50, batch_size=256, print_every=50)

    # Check loss decreased
    assert len(losses) == 50
    assert losses[-1] < losses[0], "Loss should decrease during training"
    print(f"✓ Training convergence: loss {losses[0]:.4f} → {losses[-1]:.4f}")


def test_psnr_calculation():
    """Test PSNR calculation."""
    mse_values = [0.1, 0.01, 0.001, 0.0001]
    for mse in mse_values:
        psnr = calculate_psnr(mse)
        assert psnr > 0
    print(f"✓ PSNR calculation: working")


def test_train_on_real_image():
    """Test training on actual image (short run)."""
    image_path = Path(__file__).parent.parent / "test_image.jpg"
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Prepare data
    coords, rgb, (H, W) = prepare_training_data(image_path, device, max_size=64)

    # Create small model for fast testing
    model = INRNetwork(num_frequencies=6, hidden_dim=128, num_hidden_layers=2).to(device)

    # Train briefly
    losses = train_inr(
        model, coords, rgb,
        num_epochs=100,
        batch_size=1024,
        learning_rate=1e-3,
        print_every=100
    )

    # Check training worked
    assert len(losses) == 100
    assert losses[-1] < losses[0]
    final_psnr = calculate_psnr(losses[-1])
    print(f"✓ Real image training: {H}×{W} image, final PSNR: {final_psnr:.2f} dB")


if __name__ == "__main__":
    print("=" * 70)
    print("Training Tests")
    print("=" * 70)

    tests = [
        test_training_convergence,
        test_psnr_calculation,
        test_train_on_real_image,
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
