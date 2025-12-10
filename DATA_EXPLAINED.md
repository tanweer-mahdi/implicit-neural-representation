# Data Preparation - Code Explanation

This document explains the `data.py` module, which converts images into the format needed for INR training.

---

## Overview: What Problem Are We Solving?

**Goal:** Train a network to learn `f(x, y) → (R, G, B)`

**Challenge:** Neural networks train on batches of (input, target) pairs. We need to convert an image into these pairs.

**Solution:**
- Input: 2D coordinates (x, y) for each pixel
- Target: RGB color at that pixel
- Result: Dataset of (coordinate, color) pairs

**Example:** For a 256×256 image:
- Traditional storage: 256×256×3 = 196,608 RGB values in a grid
- INR format: 65,536 coordinate pairs + 65,536 RGB values

---

## The Three Functions

### 1. `load_image()` - Load and Preprocess Image

```python
def load_image(image_path: Union[str, Path], max_size: int = 256) -> torch.Tensor:
```

**What it does:**
1. Opens image file
2. Converts to RGB (handles grayscale, RGBA, etc.)
3. Resizes if too large (maintains aspect ratio)
4. Converts to PyTorch tensor
5. Normalizes values from [0, 255] → [0, 1]

**Step-by-step breakdown:**

```python
image = Image.open(image_path).convert('RGB')
```
- Opens image using PIL
- `.convert('RGB')` ensures 3 channels (handles any format)

```python
if max(image.size) > max_size:
    image.thumbnail((max_size, max_size), Image.LANCZOS)
```
- If image is larger than `max_size`, resize it
- `.thumbnail()` maintains aspect ratio (doesn't distort)
- `Image.LANCZOS` is high-quality downsampling filter
- Example: 1024×768 with max_size=256 → 256×192

```python
image_array = torch.tensor(list(image.getdata()), dtype=torch.float32) / 255.0
```
- `image.getdata()` returns flat list of all pixels
- Convert to tensor and normalize to [0, 1]
- Why divide by 255? RGB values are 0-255, but networks work better with [0, 1]

```python
image_tensor = image_array.reshape(image.size[1], image.size[0], 3)
```
- Reshape flat list back to (H, W, 3)
- Note: `image.size` is (W, H) in PIL, we want (H, W, 3)

**Output:** Tensor of shape (H, W, 3) with values in [0, 1]

---

### 2. `create_coordinate_grid()` - Generate Pixel Coordinates

```python
def create_coordinate_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
```

**What it does:** Creates normalized (x, y) coordinates for every pixel position.

**Step-by-step breakdown:**

```python
y = torch.linspace(-1, 1, height, device=device)
x = torch.linspace(-1, 1, width, device=device)
```
- Creates evenly-spaced values from -1 to 1
- For height=256: [-1.0, -0.992, -0.984, ..., 0.984, 0.992, 1.0]
- For width=256: Same thing

**Why [-1, 1]?**
- Centered at origin (0, 0)
- Symmetric range
- Standard for positional encoding
- Makes coordinates "scale-invariant" (256×256 and 512×512 use same range)

```python
yy, xx = torch.meshgrid(y, x, indexing='ij')
```
- Creates 2D grids of coordinates
- `yy`: Each row has the same y value, varies down rows
- `xx`: Each column has the same x value, varies across columns

**Visualization for 3×3 grid:**
```
yy:               xx:
[-1  -1  -1]      [-1   0   1]
[ 0   0   0]      [-1   0   1]
[ 1   1   1]      [-1   0   1]
```

```python
coords = torch.stack([xx, yy], dim=-1)
coords = coords.reshape(-1, 2)
```
- Stack x and y into (x, y) pairs
- Reshape from (H, W, 2) to (H*W, 2)
- Each row is one pixel's coordinate

**Output:** Tensor of shape (H*W, 2) with values in [-1, 1]

**Example for 256×256:**
```
coords[0] = [-1.0, -1.0]      # Top-left pixel
coords[128] = [-0.5, -1.0]    # Somewhere on top row
coords[32768] = [0.0, 0.0]    # Center pixel
coords[65535] = [1.0, 1.0]    # Bottom-right pixel
```

---

### 3. `prepare_training_data()` - Complete Pipeline

```python
def prepare_training_data(
    image_path: Union[str, Path],
    device: torch.device,
    max_size: int = 256
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
```

**What it does:** Combines the above two functions into a complete data preparation pipeline.

**Step-by-step breakdown:**

```python
image = load_image(image_path, max_size)
H, W = image.shape[:2]
```
- Load and preprocess image
- Get dimensions (height, width)

```python
coords = create_coordinate_grid(H, W, device)
```
- Generate coordinates for every pixel
- Already on the correct device (GPU/CPU)

```python
rgb = image.reshape(-1, 3).to(device)
```
- Flatten image from (H, W, 3) to (H*W, 3)
- Move to device
- Now each row is one pixel's RGB value

**Output:**
- `coords`: (N, 2) where N = H×W
- `rgb`: (N, 3) where N = H×W
- `(H, W)`: Original image dimensions

**Example for 256×256 image:**
```
coords.shape = (65536, 2)
rgb.shape = (65536, 3)

coords[0] = [-1.0, -1.0],  rgb[0] = [0.234, 0.512, 0.789]  # Top-left pixel
coords[1] = [-0.992, -1.0], rgb[1] = [0.245, 0.523, 0.801]  # Next pixel
...
coords[65535] = [1.0, 1.0],  rgb[65535] = [0.123, 0.456, 0.678]  # Bottom-right
```

---

## Complete Example Flow

Let's trace a 4×4 image through the entire pipeline:

### Input Image (4×4 pixels):
```
Pixel Grid (visualized as colors):
[Red    Green  Blue   White ]
[Yellow Cyan   Purple Black ]
[Orange Pink   Brown  Gray  ]
[Navy   Lime   Gold   Silver]
```

### Step 1: `load_image()`
```
Output: (4, 4, 3) tensor
[[0.8, 0.0, 0.0], [0.0, 0.8, 0.0], [0.0, 0.0, 0.8], [1.0, 1.0, 1.0]]  # Row 1
[[0.8, 0.8, 0.0], [0.0, 0.8, 0.8], [0.5, 0.0, 0.5], [0.0, 0.0, 0.0]]  # Row 2
[[1.0, 0.5, 0.0], [1.0, 0.5, 0.5], [0.5, 0.3, 0.1], [0.5, 0.5, 0.5]]  # Row 3
[[0.0, 0.0, 0.5], [0.5, 1.0, 0.0], [1.0, 0.8, 0.0], [0.8, 0.8, 0.8]]  # Row 4
```

### Step 2: `create_coordinate_grid(4, 4, device)`
```
Output: (16, 2) tensor of coordinates
[[-1.0, -1.0], [-0.33, -1.0], [0.33, -1.0], [1.0, -1.0],   # Row 1 coords
 [-1.0, -0.33], [-0.33, -0.33], [0.33, -0.33], [1.0, -0.33],  # Row 2 coords
 [-1.0, 0.33], [-0.33, 0.33], [0.33, 0.33], [1.0, 0.33],   # Row 3 coords
 [-1.0, 1.0], [-0.33, 1.0], [0.33, 1.0], [1.0, 1.0]]     # Row 4 coords
```

### Step 3: Flatten RGB to (16, 3)
```
[[0.8, 0.0, 0.0],  # Red pixel
 [0.0, 0.8, 0.0],  # Green pixel
 [0.0, 0.0, 0.8],  # Blue pixel
 ... (13 more pixels)
 [0.8, 0.8, 0.8]]  # Silver pixel
```

### Final Training Data:
```
16 coordinate-RGB pairs:
coord[0] = [-1.0, -1.0]    rgb[0] = [0.8, 0.0, 0.0]  (Red at top-left)
coord[1] = [-0.33, -1.0]   rgb[1] = [0.0, 0.8, 0.0]  (Green)
coord[2] = [0.33, -1.0]    rgb[2] = [0.0, 0.0, 0.8]  (Blue)
...
coord[15] = [1.0, 1.0]     rgb[15] = [0.8, 0.8, 0.8] (Silver at bottom-right)
```

---

## Key Design Decisions

### 1. Why Normalize Coordinates to [-1, 1]?

**Options considered:**
- [0, 1]: 0 to 1 range
- [0, H] and [0, W]: Pixel indices
- [-1, 1]: Centered at origin ✓

**Why [-1, 1]?**
- **Centered:** (0, 0) is the image center, not a corner
- **Symmetric:** Same magnitude in all directions
- **Scale-invariant:** Works for any image size
- **Standard:** Used in NeRF, SDF, most INR work

---

### 2. Why Normalize RGB to [0, 1]?

**Why not keep [0, 255]?**
- Neural networks work better with values near 0
- Gradients are more stable
- Loss values are more interpretable
- Standard practice in deep learning

---

### 3. Why Use `torch.meshgrid` with `indexing='ij'`?

**Two indexing modes:**
- `'xy'`: Cartesian (x, y) - x varies along first axis
- `'ij'`: Matrix (i, j) - i varies along first axis ✓

**We use `'ij'`** because:
- Consistent with image indexing [row, col]
- Matches tensor dimensions (H, W)
- Standard for images in PyTorch

---

## Usage Example

```python
from src.inr import prepare_training_data
import torch

# Prepare data
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
coords, rgb, (H, W) = prepare_training_data('test_image.jpg', device)

print(f"Image: {H}×{W}")
print(f"Data: {coords.shape[0]:,} coordinate-RGB pairs")
print(f"Coords range: [{coords.min():.2f}, {coords.max():.2f}]")
print(f"RGB range: [{rgb.min():.2f}, {rgb.max():.2f}]")

# Now ready for training!
# Each training iteration samples random indices:
indices = torch.randint(0, len(coords), (4096,))
batch_coords = coords[indices]
batch_rgb = rgb[indices]
```

---

## Summary

**Three simple functions:**
1. `load_image()`: Image file → (H, W, 3) tensor in [0, 1]
2. `create_coordinate_grid()`: Dimensions → (H*W, 2) coords in [-1, 1]
3. `prepare_training_data()`: Combines above → training-ready data

**Key transformations:**
- Image grid → Flat list of pixels
- Pixel positions → Normalized coordinates
- RGB values → Floating-point [0, 1]

**Result:** Dataset where each pixel becomes a (coordinate, color) pair, ready for training the INR network.
