# Positional Encoding - Code Explanation

This document explains the `positional_encoding.py` implementation in detail, breaking down each major section.

---

## Overview: What Problem Are We Solving?

**Problem:** If you feed raw (x, y) coordinates like `(0.5, 0.3)` directly into a neural network, it produces blurry results. This happens because MLPs have "spectral bias" - they naturally learn smooth, low-frequency functions first and struggle with sharp edges and fine details.

**Solution:** Transform coordinates through sinusoidal functions at multiple frequencies BEFORE feeding them to the network. This creates a higher-dimensional representation where high-frequency patterns become learnable.

**Analogy:** It's like taking a 2D coordinate and expanding it into a 42D "fingerprint" that encodes position information at multiple scales.

---

## The Mathematical Formula

For a coordinate `p` (like x or y), we compute:

```
γ(p) = [p, sin(2⁰πp), cos(2⁰πp), sin(2¹πp), cos(2¹πp), ..., sin(2⁹πp), cos(2⁹πp)]
```

For 2D coordinates `(x, y)` with L=10 frequencies:
- Apply this to both x and y
- Result: 2 (original) + 2×2×10 (encoding) = **42 dimensions**

---

## Code Breakdown

### Part 1: Imports and Documentation (Lines 1-22)

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
```

**What this does:** Imports PyTorch for neural network building, numpy for constants like π.

**Key insight:** We're building a PyTorch `nn.Module` so it integrates seamlessly into neural networks.

---

### Part 2: Class Definition and __init__ (Lines 24-92)

```python
class PositionalEncoding(nn.Module):
    def __init__(
        self,
        num_frequencies: int = 10,
        include_input: bool = True,
        log_sampling: bool = True,
        scale: float = 1.0
    ):
```

**Parameters explained:**

1. **`num_frequencies` (L=10):** How many frequency bands to use
   - L=5: Fewer dimensions, captures coarse details
   - L=10: Standard, good balance (recommended)
   - L=15: More dimensions, captures very fine details

2. **`include_input` (True):** Whether to keep original (x, y) in the output
   - `True`: Output = [x, y, encodings] → 42D for 2D input
   - `False`: Output = [encodings only] → 40D for 2D input
   - Keep this `True` (helps network learn coarse structure)

3. **`log_sampling` (True):** How to space the frequencies
   - `True`: Exponential spacing → [2⁰, 2¹, 2², ..., 2⁹] → [1, 2, 4, 8, ..., 512]
   - `False`: Linear spacing → [1, 57.4, 113.8, ..., 512]
   - Keep this `True` (standard in NeRF and most work)

4. **`scale` (1.0):** Multiplier for frequencies
   - Use 1.0 if your coordinates are in [-1, 1]
   - Adjust if using different ranges

---

### Part 3: Computing Frequency Bands (Lines 72-85)

```python
if log_sampling:
    freq_bands = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)
else:
    freq_bands = torch.linspace(1.0, 2.0 ** (num_frequencies - 1), num_frequencies)

freq_bands = freq_bands * np.pi * scale
```

**What this does:**

1. **Creates frequency values:** For L=10:
   ```
   2^0 = 1, 2^1 = 2, 2^2 = 4, ..., 2^9 = 512
   ```

2. **Multiplies by π and scale:**
   ```
   [1π, 2π, 4π, 8π, 16π, 32π, 64π, 128π, 256π, 512π]
   [3.14, 6.28, 12.57, 25.13, 50.27, 100.53, 201.06, 402.12, 804.25, 1608.50]
   ```

3. **Why these specific frequencies?**
   - **Low frequencies (1π, 2π):** Capture smooth gradients (like the overall color of sky)
   - **Mid frequencies (16π, 32π):** Capture medium details (like object boundaries)
   - **High frequencies (256π, 512π):** Capture fine details (like sharp edges, textures)

**Example visualization:**
```
sin(1π × 0.5) → captures slow variations across the image
sin(512π × 0.5) → captures rapid variations (sharp edges)
```

---

### Part 4: register_buffer (Lines 87-88)

```python
self.register_buffer('freq_bands', freq_bands)
```

**What is `register_buffer`?**

This is a PyTorch-specific concept. It marks `freq_bands` as:
- **Non-trainable:** These frequencies are fixed, not learned during training
- **Part of the model state:** Moves with the model when you call `.to('mps')` or `.to('cuda')`
- **Not a parameter:** Won't show up in `model.parameters()` or get updated by optimizer

**Why not just a regular variable?**
```python
# ❌ Bad - won't move to GPU with model
self.freq_bands = freq_bands

# ✅ Good - moves to GPU automatically
self.register_buffer('freq_bands', freq_bands)
```

When you do:
```python
encoder = PositionalEncoding()
encoder = encoder.to('mps')  # freq_bands automatically moves to MPS too!
```

---

### Part 5: The Forward Pass - Core Encoding Logic (Lines 90-143)

This is where the actual encoding happens. Let's break it down step by step.

#### Step 5.1: Input Shape (Lines 123-124)

```python
def forward(self, coords: torch.Tensor) -> torch.Tensor:
    batch_size, input_dim = coords.shape
```

**Input:** `coords` with shape `(batch_size, input_dim)`
- Example: `(256, 2)` means 256 coordinate pairs (x, y)
- `batch_size=256`, `input_dim=2`

---

#### Step 5.2: Expanding Dimensions for Broadcasting (Lines 127-130)

```python
coords_expanded = coords.unsqueeze(-1)  # (batch, input_dim, 1)
freq_bands_expanded = self.freq_bands.unsqueeze(0).unsqueeze(0)  # (1, 1, num_freq)
```

**What is unsqueeze?** Adds a dimension of size 1.

**Before:**
```python
coords: (256, 2)          # 256 points, each with (x, y)
freq_bands: (10,)         # 10 frequency values
```

**After unsqueeze:**
```python
coords_expanded: (256, 2, 1)      # Added dimension at the end
freq_bands_expanded: (1, 1, 10)   # Added dimensions at the start
```

**Why do this?** To enable **broadcasting** - PyTorch's automatic dimension matching.

---

#### Step 5.3: Broadcasting and Multiplication (Lines 132-133)

```python
scaled_coords = coords_expanded * freq_bands_expanded
```

**Broadcasting magic:**
```
coords_expanded:      (256, 2, 1)
freq_bands_expanded:  (  1, 1, 10)
                      ─────────────
Result:               (256, 2, 10)
```

**What actually happens:**
```python
# For each of 256 points
# For each of 2 dimensions (x and y)
# For each of 10 frequencies
scaled_coords[i, j, k] = coords[i, j] * freq_bands[k]
```

**Concrete example:** For point (0.5, -0.3):
```
x = 0.5 gets multiplied by [3.14, 6.28, 12.57, ..., 1608.50]
  → [1.57, 3.14, 6.28, ..., 804.25]

y = -0.3 gets multiplied by [3.14, 6.28, 12.57, ..., 1608.50]
  → [-0.94, -1.88, -3.77, ..., -482.55]
```

Result shape: `(256, 2, 10)` - Each coordinate multiplied by each frequency!

---

#### Step 5.4: Apply Sin and Cos (Lines 135-137)

```python
sin_features = torch.sin(scaled_coords)  # (batch, input_dim, num_freq)
cos_features = torch.cos(scaled_coords)  # (batch, input_dim, num_freq)
```

**What this does:** Computes sin and cos of every scaled coordinate.

**Example:** For x=0.5 at frequency 2⁰π:
```
scaled_coord = 0.5 × 3.14 = 1.57
sin(1.57) ≈ 1.0
cos(1.57) ≈ 0.0
```

**Why both sin AND cos?**
- They are 90° out of phase
- Together they uniquely identify positions
- Similar to how GPS uses multiple satellites for unique positioning

**Shape:** Both are `(256, 2, 10)`

---

#### Step 5.5: Interleave Sin and Cos (Lines 139-141)

```python
encoded = torch.stack([sin_features, cos_features], dim=-1)  # (batch, input_dim, num_freq, 2)
encoded = encoded.reshape(batch_size, input_dim * self.num_frequencies * 2)
```

**Step 1 - Stack:** Puts sin and cos together:
```python
Shape: (256, 2, 10, 2)
       └──┘ └┘ └─┘ └┘
        │   │   │   └── 2 values: [sin, cos]
        │   │   └────── 10 frequencies
        │   └────────── 2 dimensions (x, y)
        └────────────── 256 points
```

**Step 2 - Reshape:** Flattens to a single vector per point:
```python
Shape: (256, 40)
       └──┘ └─┘
        │    └── 2 dimensions × 10 frequencies × 2 (sin,cos) = 40
        └────── 256 points
```

**Resulting order for each point:**
```
[sin(f0×x), cos(f0×x), sin(f1×x), cos(f1×x), ..., sin(f9×x), cos(f9×x),
 sin(f0×y), cos(f0×y), sin(f1×y), cos(f1×y), ..., sin(f9×y), cos(f9×y)]
```

---

#### Step 5.6: Optionally Include Original Input (Lines 143-144)

```python
if self.include_input:
    encoded = torch.cat([coords, encoded], dim=-1)
```

**If include_input=True:** Prepend the original coordinates
```
Before: [encodings...] → shape (256, 40)
After:  [x, y, encodings...] → shape (256, 42)
```

**Why include original input?**
- Helps the network learn coarse structure
- The original (x, y) captures "where" in broad strokes
- The encodings capture fine details

**Final output shape:** `(256, 42)` for 2D input with L=10 and include_input=True

---

### Part 6: Output Dimension Property (Lines 146-161)

```python
@property
def output_dim(self) -> int:
    return self._calculate_output_dim(input_dim=2)

def _calculate_output_dim(self, input_dim: int) -> int:
    encoding_dim = input_dim * 2 * self.num_frequencies
    if self.include_input:
        encoding_dim += input_dim
    return encoding_dim
```

**What this does:** Calculates the output dimension.

**Formula:**
```
encoding_dim = input_dim × 2 × num_frequencies
if include_input:
    encoding_dim += input_dim
```

**Example for 2D with L=10, include_input=True:**
```
encoding_dim = 2 × 2 × 10 = 40
encoding_dim += 2 = 42
```

**Why have this?** So you know what size to make the next layer:
```python
encoder = PositionalEncoding(num_frequencies=10)
next_layer = nn.Linear(encoder.output_dim, 256)  # 42 → 256
```

---

### Part 7: Functional Interface (Lines 172-206)

```python
def positional_encode(
    coords: torch.Tensor,
    num_frequencies: int = 10,
    include_input: bool = True,
    log_sampling: bool = True,
    scale: float = 1.0
) -> torch.Tensor:
    encoder = PositionalEncoding(
        num_frequencies=num_frequencies,
        include_input=include_input,
        log_sampling=log_sampling,
        scale=scale
    )
    return encoder(coords)
```

**What this is:** A convenience function for one-off encoding.

**When to use:**
- **Use the class:** When building a network (encoder reused many times)
  ```python
  encoder = PositionalEncoding(num_frequencies=10)
  for batch in dataloader:
      encoded = encoder(batch)  # Reuses same encoder
  ```

- **Use the function:** For quick experiments
  ```python
  encoded = positional_encode(coords, num_frequencies=10)  # One-off
  ```

---

### Part 8: Dimension Calculator (Lines 208-239)

```python
def get_encoding_dim(
    input_dim: int,
    num_frequencies: int = 10,
    include_input: bool = True
) -> int:
    encoding_dim = input_dim * 2 * num_frequencies
    if include_input:
        encoding_dim += input_dim
    return encoding_dim
```

**What this is:** Utility to calculate output dimension without creating an encoder.

**Usage:**
```python
# Planning network architecture
dim = get_encoding_dim(input_dim=2, num_frequencies=10, include_input=True)
print(dim)  # 42

# Now I know my network needs:
# Input: 42 dimensions (from encoder)
# Hidden: 256 dimensions
# Output: 3 dimensions (RGB)
```

---

## Complete Flow Example

Let's trace a single coordinate `(0.5, -0.3)` through the entire encoding:

### Input
```python
coords = torch.tensor([[0.5, -0.3]])  # Shape: (1, 2)
encoder = PositionalEncoding(num_frequencies=10, include_input=True)
```

### Step 1: Frequency bands (already computed in __init__)
```
freq_bands = [3.14, 6.28, 12.57, 25.13, 50.27, 100.53, 201.06, 402.12, 804.25, 1608.50]
```

### Step 2: Multiply coordinates by frequencies
```
For x = 0.5:
  0.5 × [3.14, 6.28, ...] = [1.57, 3.14, 6.28, ...]

For y = -0.3:
  -0.3 × [3.14, 6.28, ...] = [-0.94, -1.88, -3.77, ...]
```

### Step 3: Apply sin and cos
```
For x = 0.5 at each frequency:
  sin([1.57, 3.14, ...]) = [1.0, 0.0, -1.0, ...]
  cos([1.57, 3.14, ...]) = [0.0, -1.0, 0.0, ...]

For y = -0.3 at each frequency:
  sin([-0.94, -1.88, ...]) = [-0.81, -0.95, ...]
  cos([-0.94, -1.88, ...]) = [0.59, 0.31, ...]
```

### Step 4: Interleave and flatten
```
[sin(f0×x), cos(f0×x), sin(f1×x), cos(f1×x), ...,  # 20 values for x
 sin(f0×y), cos(f0×y), sin(f1×y), cos(f1×y), ...]  # 20 values for y
```

### Step 5: Prepend original coordinates
```
[0.5, -0.3, sin(f0×x), cos(f0×x), ...]  # 2 + 40 = 42 values total
```

### Output
```python
encoded = encoder(coords)
print(encoded.shape)  # torch.Size([1, 42])
```

---

## Key Insights

### 1. Why does this solve spectral bias?

**Without encoding:**
```
Network sees: (0.5, -0.3)
Can only learn smooth functions of x and y
```

**With encoding:**
```
Network sees: [0.5, -0.3, sin(1π×0.5), cos(1π×0.5), ..., sin(512π×0.5), cos(512π×0.5), ...]
Has access to high-frequency features!
```

The high-frequency sin/cos terms oscillate rapidly, giving the network "vocabulary" to learn sharp edges.

### 2. Connection to Fourier Series

This is essentially a **Fourier feature mapping**. In signal processing:
- Any function can be approximated by a sum of sines and cosines at different frequencies
- We're giving the network pre-computed basis functions (the sin/cos terms)
- The network learns the right combination (weights) to reconstruct the image

### 3. Why sin AND cos?

Consider a 1D case at position x:
- `sin(ωx)` alone: Same value at multiple positions (periodic)
- `cos(ωx)` alone: Same issue
- **Both together:** Uniquely identify position (like 2D coordinates on a circle)

### 4. Dimension explosion is intentional

2D → 42D seems extreme, but:
- **Low dimensional:** 2D coordinates are ambiguous for sharp features
- **High dimensional:** 42D gives the network rich information at multiple scales
- Trade-off: More dimensions = more parameters to learn, but better representation

---

## Common Questions

### Q: Why L=10 specifically?
**A:** Empirically found to work well for 256×256 images. Smaller images might use L=6-8, larger images might use L=12-15.

### Q: Can I skip certain frequencies?
**A:** Yes! You could use only [2⁰, 2², 2⁴] (skip odd powers). Experiment with what works for your data.

### Q: Does this make the network slower?
**A:** Slightly - the encoding adds computation. But the benefit (learning high-frequency details) far outweighs the cost.

### Q: Why not learn the frequencies?
**A:** You could! But fixed frequencies work well and are simpler. Some advanced methods (SIREN) do learn frequency-like parameters.

---

## Summary

**What the code does:**
1. Takes 2D coordinates `(x, y)`
2. Multiplies them by 10 different frequencies (1π to 512π)
3. Computes sin and cos of each
4. Concatenates everything into a 42D vector

**Why it matters:**
- Solves the spectral bias problem
- Enables learning of high-frequency details (sharp edges, textures)
- Foundation for NeRF and other neural field representations

**Key PyTorch concepts:**
- `nn.Module` structure
- `register_buffer` for non-trainable state
- Broadcasting for efficient computation
- Shape manipulation (unsqueeze, reshape, stack, cat)
