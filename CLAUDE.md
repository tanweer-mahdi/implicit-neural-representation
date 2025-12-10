# INR Implementation Guide - CLAUDE.md

## Project Overview
Building a PyTorch-based Implicit Neural Representation (INR) for images from scratch. The goal is to train a neural network to memorize an image as a continuous function rather than storing it as discrete pixels.

## Learning Objectives
1. Understand how neural networks can represent continuous signals
2. Master positional encoding and why it's necessary (spectral bias)
3. Recalibrate PyTorch fundamentals through hands-on implementation
4. Explore resolution independence and neural compression
5. Connect concepts to broader ML (NeRF, Transformers)

## Key Concepts to Understand

### Core Idea
- Traditional: Image = discrete grid of pixels at fixed coordinates
- INR: Image = continuous function f(x, y) → (R, G, B) learned by a neural network
- Goal: **Intentionally overfit** a network to memorize one image

### Spectral Bias Problem
- Raw (x, y) coordinates fed to MLPs produce blurry results
- MLPs naturally learn low-frequency functions first
- High-frequency details (sharp edges, textures) are hard to learn

### Solution: Positional Encoding
- Map coordinates through sinusoids at multiple frequencies
- γ(p) = [sin(2⁰πp), cos(2⁰πp), sin(2¹πp), cos(2¹πp), ..., sin(2^(L-1)πp), cos(2^(L-1)πp)]
- Lifts input to higher-dimensional space where high-frequency patterns become learnable
- Same concept as Transformer positional encodings!

### Resolution Independence
- Train on 256×256, render at ANY resolution (512×512, 1024×1024, even 0.5×)
- Network learns continuous color field, not discrete pixels
- Query at ANY coordinate (0.5, 0.5) or (0.3217, -0.8891)—all valid

## Implementation Plan

### Step 1: Environment Setup
**Goal**: Get PyTorch environment ready and verify setup

**Tasks**:
- Create virtual environment (optional but recommended)
- Install dependencies: `torch`, `torchvision`, `matplotlib`, `pillow`, `numpy`
- Test imports and check device (CUDA/MPS/CPU)
- Download or select a test image (256×256 or so)

**Checkpoint**: Print device type and verify image loads

**Hands-on focus**: Basic PyTorch setup, device management

---

### Step 2: Positional Encoding Module
**Goal**: Build the encoding layer that solves spectral bias

**Tasks**:
- Create `PositionalEncoding` class inheriting from `nn.Module`
- In `__init__`:
  - Store `num_frequencies` (default 10)
  - Compute frequency bands: [2^0 * π, 2^1 * π, ..., 2^(L-1) * π]
  - Use `register_buffer` for frequencies (non-trainable tensor)
- In `forward`:
  - Input shape: (batch, 2) for (x, y) coordinates
  - For each frequency: compute sin and cos
  - Concatenate all [sin, cos] pairs across frequencies and dimensions
  - Optionally include original input
  - Output shape: (batch, 2 + 2×2×L) = (batch, 42) for L=10
- Add `output_dim` helper method

**Test**:
```python
encoder = PositionalEncoding(num_frequencies=10)
sample = torch.tensor([[0.5, -0.3], [0.0, 0.0]])
encoded = encoder(sample)
print(encoded.shape)  # Should be (2, 42)
print(encoded[0, :6])  # Inspect first few values
```

**Checkpoint**: Encoding produces correct shape, values look reasonable

**Hands-on focus**:
- `nn.Module` structure
- `register_buffer` vs parameters
- Tensor broadcasting and concatenation
- Understanding shape transformations

---

### Step 3: MLP Network Architecture
**Goal**: Build the main network that maps encoded coordinates to RGB

**Tasks**:
- Create `INRNetwork` class inheriting from `nn.Module`
- In `__init__`:
  - Instantiate `PositionalEncoding`
  - Get input dimension from encoder
  - Build MLP with `nn.Sequential`:
    - Input: Linear(input_dim, 256) + ReLU
    - Hidden: 4 layers of Linear(256, 256) + ReLU
    - Output: Linear(256, 3) + Sigmoid
- In `forward`:
  - Apply positional encoding to coordinates
  - Pass through MLP
  - Return RGB predictions in [0, 1]

**Test**:
```python
model = INRNetwork(num_frequencies=10, hidden_dim=256, num_hidden_layers=4)
coords = torch.rand(100, 2) * 2 - 1  # Random in [-1, 1]
output = model(coords)
print(output.shape)  # (100, 3)
print(output.min(), output.max())  # Should be in [0, 1]
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
```

**Checkpoint**: Network forward pass works, outputs in valid range

**Hands-on focus**:
- Building MLPs with `nn.Sequential`
- Activation functions (ReLU vs alternatives)
- Output constraints (Sigmoid for [0, 1] range)
- Counting parameters

---

### Step 4: Data Preparation
**Goal**: Convert image into coordinate-RGB pairs for training

**Tasks**:
- Implement `load_image(path, max_size=256)`:
  - Load with PIL
  - Convert to RGB
  - Resize if needed (maintain aspect ratio)
  - Convert to torch tensor, normalize to [0, 1]
  - Shape: (H, W, 3)

- Implement `create_coordinate_grid(height, width, device)`:
  - Use `torch.linspace(-1, 1, height)` for y
  - Use `torch.linspace(-1, 1, width)` for x
  - Create meshgrid with `torch.meshgrid`
  - Stack into (H, W, 2) then reshape to (H*W, 2)
  - Return normalized coordinates

- Implement `prepare_training_data(image, device)`:
  - Get H, W from image
  - Create coordinate grid
  - Flatten image to (H*W, 3)
  - Move both to device
  - Return coords, rgb tensors

**Test**:
```python
image = load_image("test_image.jpg", max_size=256)
print(f"Image shape: {image.shape}")
coords, rgb = prepare_training_data(image, device)
print(f"Coords: {coords.shape}, range [{coords.min():.2f}, {coords.max():.2f}]")
print(f"RGB: {rgb.shape}, range [{rgb.min():.2f}, {rgb.max():.2f}]")
```

**Checkpoint**: Image loads correctly, coordinates in [-1, 1], RGB in [0, 1]

**Hands-on focus**:
- Image loading and preprocessing
- Coordinate grid creation
- Tensor reshaping and flattening
- Device management

---

### Step 5: Training Loop
**Goal**: Train network to memorize the image

**Tasks**:
- Implement `train(model, coords, rgb, num_epochs=2000, batch_size=4096, lr=1e-3)`:
  - Create Adam optimizer with given learning rate
  - Use MSELoss criterion
  - For each epoch:
    - Sample random batch indices: `torch.randint(0, len(coords), (batch_size,))`
    - Get batch coords and RGB
    - Zero gradients
    - Forward pass: `pred_rgb = model(batch_coords)`
    - Compute loss: `loss = criterion(pred_rgb, batch_rgb)`
    - Backward pass: `loss.backward()`
    - Optimizer step
    - Record loss
    - Every 100 epochs: print epoch, loss, PSNR
  - Return loss history

- Add PSNR calculation helper:
  ```python
  def calculate_psnr(mse):
      return -10 * torch.log10(mse)
  ```

**Test**:
```python
model = INRNetwork().to(device)
losses = train(model, coords, rgb, num_epochs=500, batch_size=4096)
print(f"Initial loss: {losses[0]:.6f}")
print(f"Final loss: {losses[-1]:.6f}")
plt.plot(losses)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.show()
```

**Checkpoint**: Loss decreases from ~0.1 to <0.001, training converges

**Hands-on focus**:
- Training loop structure
- Gradient computation and backpropagation
- Optimizer mechanics
- Random batch sampling
- Loss monitoring

---

### Step 6: Inference and Rendering
**Goal**: Render the learned image at any resolution

**Tasks**:
- Implement `render(model, height, width, device, batch_size=8192)`:
  - Set model to eval mode: `model.eval()`
  - Use `@torch.no_grad()` decorator
  - Create coordinate grid at desired resolution
  - Process in batches to avoid OOM:
    - Split coords into batches
    - Forward pass each batch
    - Collect predictions
  - Concatenate all predictions
  - Reshape to (H, W, 3)
  - Clamp to [0, 1], convert to numpy
  - Scale to [0, 255] and convert to uint8
  - Return as numpy array

**Test**:
```python
# After training
model.eval()
H, W = image.shape[:2]

# Same resolution
recon = render(model, H, W, device)
print(f"Reconstruction: {recon.shape}")

# 2× super-resolution
recon_2x = render(model, H*2, W*2, device)
print(f"2× super-res: {recon_2x.shape}")

# 4× super-resolution
recon_4x = render(model, H*4, W*4, device)
print(f"4× super-res: {recon_4x.shape}")
```

**Checkpoint**: Rendering works at multiple resolutions

**Hands-on focus**:
- Inference mode (`eval()` and `no_grad()`)
- Batch processing for memory efficiency
- Tensor to numpy conversion
- Understanding resolution independence

---

### Step 7: Visualization
**Goal**: Compare original, reconstruction, and super-resolution

**Tasks**:
- Implement `visualize_results(original, model, device, losses)`:
  - Create figure with subplots (2 rows × 3 cols recommended)
  - Plot 1: Original image
  - Plot 2: Reconstruction at same resolution
  - Plot 3: 2× super-resolution
  - Plot 4: Training loss curve (log scale)
  - Plot 5: Zoomed crop comparison (center region)
  - Plot 6: 4× super-resolution or difference map
  - Add titles, remove axes for images
  - Save figure to file

- Add MSE and PSNR calculation between original and reconstruction

**Test**:
```python
visualize_results(image, model, device, losses)
# Should produce comprehensive visualization showing:
# - Quality of reconstruction
# - Super-resolution capability
# - Training convergence
```

**Checkpoint**: Visual confirmation that network learned the image

**Hands-on focus**:
- Matplotlib visualization
- Image comparison techniques
- Understanding reconstruction quality

---

### Step 8: Main Script and Experimentation
**Goal**: Put it all together and experiment

**Tasks**:
- Create `main()` function that orchestrates Steps 1-7
- Add timing measurements
- Save model checkpoint: `torch.save(model.state_dict(), 'inr_model.pth')`
- Load model checkpoint: `model.load_state_dict(torch.load('inr_model.pth'))`

**Experiments to run**:
1. **Effect of positional encoding**:
   - Train without encoding (raw coords)
   - Train with L=5, L=10, L=15 frequencies
   - Compare reconstruction quality

2. **Architecture variations**:
   - Try 2, 4, 6 hidden layers
   - Try hidden_dim of 128, 256, 512
   - Compare parameters vs quality tradeoff

3. **Resolution tests**:
   - Train on 256×256
   - Render at 128×128, 256×256, 512×512, 1024×1024
   - Observe quality at different scales

4. **Different images**:
   - Try natural photos (smooth gradients)
   - Try graphics/cartoons (sharp edges)
   - Try textures (high frequency)
   - Observe which are harder to fit

**Checkpoint**: Complete working implementation with experiments

**Hands-on focus**:
- Model persistence
- Systematic experimentation
- Understanding hyperparameter effects
- Critical analysis of results

---

## File Structure Recommendation

### Option A: Single File (Recommended for Learning)
```
implicit-neural-representation/
├── inr_simple.py          # All code in one file
├── test_image.jpg         # Sample image
├── inr_model.pth         # Saved model weights
└── results/              # Generated visualizations
```

### Option B: Modular Structure (Better for Production)
```
implicit-neural-representation/
├── src/
│   ├── __init__.py
│   ├── positional_encoding.py  # Step 2
│   ├── network.py               # Step 3
│   ├── data_utils.py            # Step 4
│   ├── train.py                 # Step 5
│   ├── inference.py             # Step 6
│   └── visualize.py             # Step 7
├── main.py                      # Step 8 orchestration
├── experiments.py               # For trying variations
├── test_image.jpg
└── results/
```

**For hands-on learning, start with Option A (single file).** This makes it easier to understand the full flow. Refactor to Option B later if desired.

---

## Expected Results

After training for ~2000 epochs on a 256×256 image:

| Metric | Expected Value |
|--------|---------------|
| Final MSE Loss | 0.0001 to 0.001 |
| PSNR | 30-40 dB |
| Training Time (GPU) | 2-5 minutes |
| Training Time (CPU) | 10-15 minutes |
| Network Size | ~300K parameters (~1.2 MB) |
| Memory Usage | <1 GB |

**Quality**:
- Reconstruction should be visually indistinguishable from original
- Super-resolution should look smooth and natural
- Sharp edges should be preserved (with proper positional encoding)

---

## Debugging Checklist

If results are poor:

**Blurry reconstruction**:
- ✓ Positional encoding implemented correctly?
- ✓ Using enough frequencies? (Try L=10 or higher)
- ✓ Coordinates normalized to [-1, 1]?

**Not converging**:
- ✓ Learning rate appropriate? (Try 1e-3 or 1e-4)
- ✓ Batch size reasonable? (4096 works well)
- ✓ Enough epochs? (Try 2000+)
- ✓ RGB values in [0, 1] range?

**Memory issues**:
- ✓ Reduce batch size
- ✓ Use smaller image (128×128)
- ✓ Render in batches (should already be doing this)

**Artifacts or strange colors**:
- ✓ Check Sigmoid at output (should clamp to [0, 1])
- ✓ Verify image loading (PIL to tensor correctly)
- ✓ Check for NaN in loss (reduce learning rate)

---

## Key PyTorch Concepts Reinforced

1. **Module design**: Subclassing `nn.Module`, `__init__` and `forward`
2. **Buffers vs Parameters**: `register_buffer` for non-trainable tensors
3. **Training loop**: Zero grad, forward, loss, backward, step
4. **Device management**: Moving tensors and models to GPU/CPU
5. **Batch processing**: Handling variable batch sizes efficiently
6. **Inference mode**: `model.eval()` and `torch.no_grad()`
7. **Model persistence**: Saving and loading state dicts
8. **Tensor operations**: Reshaping, concatenation, meshgrid
9. **Gradient flow**: Understanding backpropagation through custom modules
10. **Optimization**: Adam optimizer, learning rates, convergence

---

## Extensions to Explore (After Core Implementation)

1. **SIREN architecture**: Replace ReLU with sin activations throughout
2. **Video INR**: Add time dimension (x, y, t) → (R, G, B)
3. **3D shapes**: Implement signed distance functions (SDF)
4. **Compression**: Compare network size to JPEG at different qualities
5. **Animation**: Interpolate between learned weights of different images
6. **Inpainting**: Mask training data, see if network hallucinates
7. **Meta-learning**: Train on multiple images, test generalization

---

## Interview Connection Points

When discussing this project:

1. **Novel approach**: "Instead of storing pixels, I trained a network to BE the image"
2. **Spectral bias**: "Learned why MLPs struggle with high frequencies and solved it with positional encoding"
3. **Transformer connection**: "Same positional encoding concept—position in sequence vs position in 2D space"
4. **Resolution independence**: "Train once, render at any resolution—it's a continuous function"
5. **NeRF foundation**: "This 2D concept extends to 3D + viewing direction = Neural Radiance Fields"
6. **Overfitting intentionally**: "One of the rare cases where memorization is the goal"
7. **Practical tradeoffs**: "Slower inference than traditional images, but unique capabilities like neural compression"

---

## Success Criteria

You'll know you've succeeded when:

✓ Network converges to <0.001 MSE loss
✓ Reconstruction is visually identical to original
✓ Can render at 4× resolution with smooth results
✓ Understand why positional encoding is necessary
✓ Can explain spectral bias clearly
✓ Comfortable with PyTorch module design
✓ Can experiment with architecture variations
✓ Can connect concepts to NeRF and Transformers

---

## Next Steps After Completion

1. Write a blog post explaining your implementation
2. Try implementing SIREN (sinusoidal activations)
3. Extend to video (add time dimension)
4. Read the original NeRF paper with deeper understanding
5. Explore Instant-NGP hash encoding for speed improvements

---

## Resources

**Papers**:
- Fourier Features Let Networks Learn High Frequency Functions (Tancik et al., 2020)
- SIREN: Implicit Neural Representations with Periodic Activation Functions (Sitzmann et al., 2020)
- NeRF: Representing Scenes as Neural Radiance Fields (Mildenhall et al., 2020)

**Code references**:
- Official NeRF implementation: github.com/bmild/nerf
- SIREN implementation: github.com/vsitzmann/siren

Good luck with your implementation! Take it step by step, test thoroughly at each stage, and don't hesitate to experiment.
