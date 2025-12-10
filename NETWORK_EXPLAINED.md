# INRNetwork - Code Explanation

This document explains the `network.py` implementation, which builds the MLP that learns to map coordinates to RGB colors.

---

## Overview: What Does This Network Do?

**Goal:** Learn a function `f(x, y) → (R, G, B)` that represents an image.

**Input:** 2D coordinates `(x, y)` in range [-1, 1]
**Output:** RGB color values in range [0, 1]

**Architecture:**
```
Coordinates (x, y)
    ↓
Positional Encoding (2D → 42D)
    ↓
Linear(42 → 256) + ReLU
    ↓
Linear(256 → 256) + ReLU  ← Hidden layer 1
    ↓
Linear(256 → 256) + ReLU  ← Hidden layer 2
    ↓
Linear(256 → 256) + ReLU  ← Hidden layer 3
    ↓
Linear(256 → 256) + ReLU  ← Hidden layer 4
    ↓
Linear(256 → 3) + Sigmoid
    ↓
RGB (R, G, B)
```

---

## Code Breakdown

### Part 1: Class Definition and __init__ (Lines 18-98)

#### Parameters (Lines 60-68)

```python
def __init__(
    self,
    num_frequencies: int = 10,
    hidden_dim: int = 256,
    num_hidden_layers: int = 4,
    include_input: bool = True,
    use_bias: bool = True
):
```

**Parameters explained:**

1. **`num_frequencies` (L=10):**
   - Passed to positional encoder
   - More frequencies = can learn finer details
   - Default 10 is good for 256×256 images

2. **`hidden_dim` (256):**
   - Width of each hidden layer (number of neurons)
   - **Bigger = more capacity** but slower training and more memory
   - 256 is a sweet spot (gives ~275K parameters)
   - Compare:
     - 128: ~75K parameters, may underfit
     - 256: ~275K parameters, good balance ✓
     - 512: ~1M parameters, may be overkill for small images

3. **`num_hidden_layers` (4):**
   - Depth of network (not counting input/output layers)
   - **More layers = more depth** to learn complex functions
   - 4 is standard, gives good results
   - Compare:
     - 2 layers: Simpler, faster, may underfit
     - 4 layers: Standard, good balance ✓
     - 8 layers: Deeper, may help with very complex images

4. **`include_input` (True):**
   - Whether to include original (x, y) in encoding
   - Keep True (helps with coarse structure)

5. **`use_bias` (True):**
   - Whether Linear layers have bias terms
   - Keep True (standard practice)

---

#### Creating the Positional Encoder (Lines 76-80)

```python
self.encoder = PositionalEncoding(
    num_frequencies=num_frequencies,
    include_input=include_input
)
```

**What this does:** Creates the positional encoder we built in Step 2.

**Result:**
- Input: 2D coordinates
- Output: 42D encoded vectors (for L=10 with include_input=True)

---

#### Calculating Input Dimension (Lines 82-83)

```python
input_dim = self.encoder._calculate_output_dim(input_dim=2)
```

**What this does:** Asks the encoder "what size will your output be?"

**Why?** So we know what size to make the first Linear layer.

**Example:**
- If L=10 and include_input=True: `input_dim = 42`
- If L=5 and include_input=True: `input_dim = 22`

---

#### Building the MLP (Lines 85-98)

This is the core of the network. We build it layer by layer.

```python
layers: List[nn.Module] = []
```

**Start with an empty list** - we'll add layers one by one, then combine them.

---

##### Input Layer (Lines 88-90)

```python
layers.append(nn.Linear(input_dim, hidden_dim, bias=use_bias))
layers.append(nn.ReLU(inplace=True))
```

**What this does:**
1. **Linear(42 → 256):** Transforms 42D encoding to 256D
   - Has 42 × 256 = 10,752 weights + 256 biases = **11,008 parameters**
2. **ReLU:** Activation function, introduces non-linearity
   - Formula: `ReLU(x) = max(0, x)`
   - Keeps positive values, zeros out negative values

**Why ReLU?**
- Simple and effective
- Prevents vanishing gradients
- Fast to compute

**The `inplace=True`:** Memory optimization - modifies tensor in-place rather than creating a copy.

---

##### Hidden Layers (Lines 92-95)

```python
for _ in range(num_hidden_layers):
    layers.append(nn.Linear(hidden_dim, hidden_dim, bias=use_bias))
    layers.append(nn.ReLU(inplace=True))
```

**What this does:** Creates 4 identical hidden layers (for default config).

**Each hidden layer:**
- Linear(256 → 256): 256 × 256 = 65,536 weights + 256 biases = **65,792 parameters**
- ReLU activation

**For 4 hidden layers:**
- Total: 4 × 65,792 = **263,168 parameters** (most of the network!)

**Why same size throughout?**
- **Simpler architecture** - easier to reason about
- **Highway for information** - no bottlenecks
- **Standard practice** for INR/NeRF

---

##### Output Layer (Lines 97-99)

```python
layers.append(nn.Linear(hidden_dim, 3, bias=use_bias))
layers.append(nn.Sigmoid())
```

**What this does:**
1. **Linear(256 → 3):** Projects to RGB
   - 256 × 3 = 768 weights + 3 biases = **771 parameters**
   - Output: 3 values (one for R, G, B)

2. **Sigmoid:** Squashes output to [0, 1]
   - Formula: `sigmoid(x) = 1 / (1 + e^(-x))`
   - Maps (-∞, +∞) → (0, 1)
   - Ensures valid RGB values

**Why Sigmoid at the end?**
- **Constraint:** RGB must be in [0, 1]
- **Alternative:** Could use Tanh (gives [-1, 1]) then scale/shift
- **Standard:** Sigmoid is typical for INR

**Total Parameters:**
- Input layer: 11,008
- Hidden layers: 263,168 (4 × 65,792)
- Output layer: 771
- **Grand total: 274,947 parameters**

---

##### Combining Layers (Line 101)

```python
self.mlp = nn.Sequential(*layers)
```

**What this does:** Combines all layers into a single Sequential module.

**`nn.Sequential`:** Runs layers in order, passing output of one as input to next.

**The `*layers` syntax:** Unpacks the list
```python
# These are equivalent:
nn.Sequential(*[layer1, layer2, layer3])
nn.Sequential(layer1, layer2, layer3)
```

**Result:** `self.mlp` is now a complete network that can be called like a function.

---

### Part 2: Forward Pass (Lines 103-121)

```python
def forward(self, coords: torch.Tensor) -> torch.Tensor:
    # Step 1: Encode coordinates (2D → 42D)
    encoded = self.encoder(coords)

    # Step 2: Pass through MLP (42D → 3D RGB)
    rgb = self.mlp(encoded)

    return rgb
```

**What this does:** Defines what happens when you call `model(coords)`.

**Flow:**
1. **Input:** `coords` with shape `(batch, 2)`
   - Example: `(256, 2)` for 256 coordinate pairs

2. **Encoding:** `encoded = self.encoder(coords)`
   - Shape: `(256, 42)`
   - Coordinates now in high-dimensional space

3. **MLP:** `rgb = self.mlp(encoded)`
   - Passes through all layers sequentially
   - Shape: `(256, 3)`
   - Each row is an RGB triplet

4. **Output:** `rgb` with shape `(batch, 3)`, values in [0, 1]

**Example with actual numbers:**
```python
coords = torch.tensor([[0.5, -0.3]])  # One point

# After encoding:
encoded = [0.5, -0.3, sin(...), cos(...), ...]  # 42 values

# After input layer + ReLU:
hidden = [0.12, 0.0, 0.87, ...]  # 256 values (some zeros from ReLU)

# ... through hidden layers ...

# After output layer + Sigmoid:
rgb = [0.73, 0.51, 0.39]  # RGB for a brownish color
```

---

### Part 3: Helper Methods (Lines 123-180)

#### count_parameters() (Lines 123-133)

```python
def count_parameters(self) -> int:
    return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

**What this does:** Counts total trainable parameters.

**How it works:**
- `self.parameters()`: Gets all parameters (weights and biases)
- `p.numel()`: Counts elements in each parameter tensor
- `p.requires_grad`: Only counts trainable parameters
- `sum(...)`: Adds them all up

**Example:**
```python
model = INRNetwork()
print(model.count_parameters())  # 274,947
```

**Why this matters:**
- **Memory:** More parameters = more GPU memory
- **Training time:** More parameters = slower training
- **Capacity:** More parameters = can fit more complex functions

---

#### get_architecture_summary() (Lines 135-169)

```python
def get_architecture_summary(self) -> str:
    encoding_dim = self.encoder._calculate_output_dim(input_dim=2)
    param_count = self.count_parameters()

    summary = [
        "INR Network Architecture",
        "=" * 60,
        f"Input: 2D coordinates (x, y) in [-1, 1]",
        # ... more lines ...
    ]
    return "\n".join(summary)
```

**What this does:** Creates a pretty-printed summary of the architecture.

**Example output:**
```
INR Network Architecture
============================================================
Input: 2D coordinates (x, y) in [-1, 1]

Positional Encoding (L=10):
  2D → 42D

MLP Architecture:
  Input layer:  Linear(42 → 256) + ReLU
  Hidden layer 1: Linear(256 → 256) + ReLU
  Hidden layer 2: Linear(256 → 256) + ReLU
  Hidden layer 3: Linear(256 → 256) + ReLU
  Hidden layer 4: Linear(256 → 256) + ReLU
  Output layer: Linear(256 → 3) + Sigmoid

Output: RGB colors in [0, 1]

Total parameters: 274,947
============================================================
```

**Usage:**
```python
model = INRNetwork()
print(model.get_architecture_summary())
```

---

### Part 4: Factory Function (Lines 183-216)

```python
def create_inr_network(
    num_frequencies: int = 10,
    hidden_dim: int = 256,
    num_hidden_layers: int = 4,
    device: Optional[torch.device] = None
) -> INRNetwork:
    model = INRNetwork(
        num_frequencies=num_frequencies,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers
    )

    if device is not None:
        model = model.to(device)

    return model
```

**What this is:** Convenience function for creating and placing a network on a device.

**Usage:**
```python
# Without factory:
model = INRNetwork(hidden_dim=256)
model = model.to('mps')

# With factory (simpler):
model = create_inr_network(hidden_dim=256, device='mps')
```

**Why have this?** Common pattern in PyTorch - factory functions simplify common workflows.

---

## Architecture Design Choices

### 1. Why ReLU instead of other activations?

**ReLU:** `f(x) = max(0, x)`
- ✅ Simple and fast
- ✅ No vanishing gradient problem
- ✅ Works well in practice
- ❌ Can cause "dying ReLU" (neurons get stuck at 0)

**Alternatives:**
- **SIREN:** Uses `sin(x)` throughout (specialized INR architecture)
- **Leaky ReLU:** `f(x) = x if x > 0 else 0.01x` (prevents dying neurons)
- **GELU:** Smooth activation used in Transformers

**Our choice:** ReLU is standard and works well for INR.

---

### 2. Why 4 hidden layers?

**Depth vs. Width trade-off:**

| Layers | Pros | Cons |
|--------|------|------|
| 2 | Fast, simple | May underfit complex images |
| 4 | Good balance ✓ | - |
| 8 | Can model very complex functions | Slower, may overfit, harder to train |

**Empirical finding:** 4 layers with 256 neurons each works well for images up to 512×512.

---

### 3. Why Sigmoid at output?

**Need:** RGB values must be in [0, 1].

**Options:**
1. **Sigmoid:** Maps to [0, 1] directly ✓
2. **Tanh + scaling:** Maps to [-1, 1], then `(tanh(x) + 1) / 2`
3. **No activation + clipping:** `torch.clamp(x, 0, 1)`

**Our choice:** Sigmoid is simplest and most common.

**Note:** The network learns to produce values that, after sigmoid, match the target RGB. The sigmoid guides training.

---

### 4. Why same hidden dimension throughout?

**Alternative:** Could use bottleneck: 42 → 512 → 256 → 128 → 3

**Our choice (constant 256):**
- ✅ Simpler to reason about
- ✅ Information flows freely (no bottlenecks)
- ✅ Standard in INR literature

**Bottleneck might help:** If you wanted to force the network to learn a compressed representation, but that's not the goal here.

---

## How the Network Learns

### Training Process (Preview for Step 5)

1. **Forward pass:**
   ```python
   rgb_predicted = model(coords)  # (N, 3)
   ```

2. **Compare to target:**
   ```python
   loss = mse_loss(rgb_predicted, rgb_actual)
   ```

3. **Backward pass:**
   ```python
   loss.backward()  # Computes gradients
   ```

4. **Update weights:**
   ```python
   optimizer.step()  # Updates all 274,947 parameters
   ```

**What the network learns:**
- **Early layers:** Extract features from positional encoding
- **Middle layers:** Combine features to detect patterns
- **Final layer:** Map patterns to specific RGB values

**Memorization vs. Generalization:**
- For INR, we **want to memorize** the specific image
- Unlike typical ML, overfitting is the goal!
- The network becomes a compressed representation of the image

---

## Parameter Count Breakdown

For standard config (L=10, hidden_dim=256, 4 hidden layers):

| Layer | Size | Parameters | Calculation |
|-------|------|------------|-------------|
| Encoding | 2 → 42 | 0 | (non-trainable) |
| Input | 42 → 256 | 11,008 | 42×256 + 256 |
| Hidden 1 | 256 → 256 | 65,792 | 256×256 + 256 |
| Hidden 2 | 256 → 256 | 65,792 | 256×256 + 256 |
| Hidden 3 | 256 → 256 | 65,792 | 256×256 + 256 |
| Hidden 4 | 256 → 256 | 65,792 | 256×256 + 256 |
| Output | 256 → 3 | 771 | 256×3 + 3 |
| **Total** | | **274,947** | |

**Memory:**
- Float32: 4 bytes per parameter
- Total: 274,947 × 4 = ~1.1 MB (just for weights!)
- Add gradients: ~2.2 MB during training
- Add optimizer state (Adam): ~4.4 MB total

---

## Usage Examples

### Basic Usage

```python
from src.inr import INRNetwork

# Create network
model = INRNetwork(num_frequencies=10, hidden_dim=256, num_hidden_layers=4)

# Forward pass
coords = torch.rand(100, 2) * 2 - 1  # 100 random coords in [-1, 1]
rgb = model(coords)

print(rgb.shape)  # torch.Size([100, 3])
print(rgb.min(), rgb.max())  # Values in [0, 1]
```

### Architecture Inspection

```python
# Count parameters
print(f"Parameters: {model.count_parameters():,}")  # 274,947

# View architecture
print(model.get_architecture_summary())

# See all layers
print(model)
```

### Different Configurations

```python
# Smaller network (faster, less capacity)
small = INRNetwork(hidden_dim=128, num_hidden_layers=2)
print(small.count_parameters())  # ~75,000

# Larger network (slower, more capacity)
large = INRNetwork(hidden_dim=512, num_hidden_layers=6)
print(large.count_parameters())  # ~1,800,000
```

---

## Key Insights

### 1. The Network IS the Image

After training:
- **Traditional:** Image stored as pixel array (256×256×3 = 196,608 values)
- **INR:** Image stored as network weights (274,947 parameters)

**Paradox:** More parameters than pixels!
- But parameters are more compact (learned compression)
- Can render at ANY resolution
- Resolution-independent representation

### 2. No Convolutions!

**Traditional CNNs:** Use convolutions to understand spatial relationships.

**INR:** No convolutions! Just fully-connected (Linear) layers.
- Spatial understanding comes from **positional encoding**
- The sin/cos features encode "position" explicitly
- MLP learns to map "position" → "color"

### 3. Sigmoid is a Soft Constraint

**Without Sigmoid:** Network could output any values (-∞ to +∞)

**With Sigmoid:** Gently guides outputs toward [0, 1]
- During training, network learns to produce pre-sigmoid values that, after sigmoid, match targets
- Acts as a "soft constraint" (not hard clipping)

---

## Common Questions

### Q: Why so many parameters for a small image?
**A:** The network is learning a continuous function, not just storing pixels. The parameters encode the function, and we need enough capacity to represent all the details.

### Q: Could we use fewer hidden layers?
**A:** Yes! Try 2 layers for simple images. But 4 layers gives good results for most natural images.

### Q: Why is the middle layer all the same size (256)?
**A:** Simplicity and consistency. Information flows freely without bottlenecks.

### Q: What if RGB goes outside [0, 1] before Sigmoid?
**A:** That's fine! Sigmoid maps any real number to [0, 1]. Values like -10 → ~0, +10 → ~1.

---

## Summary

**What we built:**
- MLP with positional encoding frontend
- 274,947 trainable parameters
- Maps 2D coordinates → RGB colors
- Clean, modular architecture

**Key components:**
1. Positional encoder (from Step 2)
2. Input layer (42D → 256D)
3. 4 hidden layers (256D → 256D each)
4. Output layer (256D → 3D RGB)

**Next step (Step 4):**
- Prepare training data (image → coordinate-RGB pairs)
- Then we can actually train this network to memorize an image!
