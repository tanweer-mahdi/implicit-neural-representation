# Implicit Neural Representation (INR) - PyTorch Learning Project

Learning INR by building it step-by-step in PyTorch.

## Project Structure

```
.
├── src/inr/
│   ├── positional_encoding.py         # Step 2: Positional encoding
│   ├── network.py                     # Step 3: MLP network
│   └── data.py                        # Step 4: Data preparation
├── tests/
│   ├── test_positional_encoding.py
│   ├── test_network.py
│   └── test_data.py
├── config.py                          # Network & training configuration
├── test_image.jpg                     # Sample image
├── POSITIONAL_ENCODING_EXPLAINED.md
├── NETWORK_EXPLAINED.md
├── DATA_EXPLAINED.md
├── CLAUDE.md                          # Step-by-step guide
└── inr_implementation.md              # Theory
```

## Steps

- ✅ Step 1: Environment Setup
- ✅ Step 2: Positional Encoding
- ✅ Step 3: MLP Network
- ✅ Step 4: Data Preparation
- ⏳ Step 5: Training Loop
- ⏳ Step 6: Inference
- ⏳ Step 7: Visualization
- ⏳ Step 8: Experiments

## Running Tests

```bash
uv run python tests/test_positional_encoding.py
uv run python tests/test_network.py
uv run python tests/test_data.py
```

## Usage

```python
from src.inr import INRNetwork

# Create network
model = INRNetwork(num_frequencies=10, hidden_dim=256, num_hidden_layers=4)

# Forward pass
coords = torch.rand(100, 2) * 2 - 1  # (batch, 2) in [-1, 1]
rgb = model(coords)                   # (batch, 3) in [0, 1]

print(f"Parameters: {model.count_parameters():,}")  # 274,947
```
