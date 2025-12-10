"""
Positional Encoding for Implicit Neural Representations

This module implements sinusoidal positional encoding to overcome the spectral bias
of MLPs. The encoding maps low-dimensional coordinates to a higher-dimensional space
where high-frequency patterns become learnable.

Key Concept:
    MLPs naturally learn low-frequency functions first (spectral bias). By encoding
    input coordinates with sinusoids at multiple frequencies, we enable the network
    to learn high-frequency details (sharp edges, fine textures).

Connection to Transformers:
    This is the SAME concept used in Transformer positional encodings! In Transformers,
    position indices are encoded with sinusoids so the model can learn position-dependent
    patterns. In INR, spatial coordinates are encoded similarly.

References:
    - Fourier Features Let Networks Learn High Frequency Functions (Tancik et al., 2020)
    - NeRF: Representing Scenes as Neural Radiance Fields (Mildenhall et al., 2020)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for coordinate inputs.

    Transforms input coordinates through sinusoids at multiple frequencies:
        γ(p) = [sin(2⁰πp), cos(2⁰πp), sin(2¹πp), cos(2¹πp), ..., sin(2^(L-1)πp), cos(2^(L-1)πp)]

    For 2D coordinates (x, y) with L frequencies:
        - Each coordinate gets encoded into 2L values (sin and cos at each frequency)
        - Total encoding dimension: input_dim × 2 × L (plus optionally the original coordinates)

    Args:
        num_frequencies (int): Number of frequency bands (L). Higher values capture finer details.
                               Default: 10 (produces 42D output for 2D input with include_input=True)
        include_input (bool): Whether to concatenate original input coordinates.
                              Recommended: True (helps with coarse structure)
                              Default: True
        log_sampling (bool): If True, frequencies are 2^k for k in [0, L-1] (exponential spacing)
                            If False, frequencies are linearly spaced.
                            Recommended: True (standard in NeRF and most INR work)
                            Default: True
        scale (float): Scaling factor for frequencies. Adjust based on input coordinate range.
                       Default: 1.0 (assumes input in [-1, 1])

    Shape:
        - Input: (batch_size, input_dim) where input_dim is typically 2 for (x, y)
        - Output: (batch_size, output_dim) where output_dim = input_dim × 2 × L (+ input_dim if include_input=True)

    Example:
        >>> encoder = PositionalEncoding(num_frequencies=10, include_input=True)
        >>> coords = torch.tensor([[0.5, -0.3], [0.0, 0.0]])  # 2 points in 2D
        >>> encoded = encoder(coords)
        >>> print(encoded.shape)  # torch.Size([2, 42])
        >>> print(encoder.output_dim)  # 42
    """

    def __init__(
        self,
        num_frequencies: int = 10,
        include_input: bool = True,
        log_sampling: bool = True,
        scale: float = 1.0
    ):
        super().__init__()

        self.num_frequencies = num_frequencies
        self.include_input = include_input
        self.log_sampling = log_sampling
        self.scale = scale

        # Precompute frequency bands
        # Standard: [2^0 * π, 2^1 * π, ..., 2^(L-1) * π]
        if log_sampling:
            # Exponential spacing: 2^k for k in [0, L-1]
            freq_bands = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)
        else:
            # Linear spacing: equally spaced from 1 to 2^(L-1)
            freq_bands = torch.linspace(1.0, 2.0 ** (num_frequencies - 1), num_frequencies)

        # Scale by π and user-defined scale factor
        freq_bands = freq_bands * np.pi * scale

        # Register as buffer (non-trainable, but moves with model to device)
        self.register_buffer('freq_bands', freq_bands)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding to input coordinates.

        Args:
            coords: Input coordinates of shape (batch_size, input_dim)
                   Typically (N, 2) for 2D coordinates (x, y)

        Returns:
            Encoded coordinates of shape (batch_size, output_dim)

        Mathematical operation:
            For each coordinate dimension p and each frequency f_k:
                output = [..., sin(f_k * p), cos(f_k * p), ...]
        """
        batch_size, input_dim = coords.shape

        # Expand dimensions for broadcasting
        # coords: (batch, input_dim) -> (batch, input_dim, 1)
        # freq_bands: (num_frequencies,) -> (1, 1, num_frequencies)
        coords_expanded = coords.unsqueeze(-1)  # (batch, input_dim, 1)
        freq_bands_expanded = self.freq_bands.unsqueeze(0).unsqueeze(0)  # (1, 1, num_freq)

        # Multiply coordinates by frequency bands
        # Result: (batch, input_dim, num_frequencies)
        scaled_coords = coords_expanded * freq_bands_expanded

        # Apply sin and cos
        sin_features = torch.sin(scaled_coords)  # (batch, input_dim, num_freq)
        cos_features = torch.cos(scaled_coords)  # (batch, input_dim, num_freq)

        # Interleave sin and cos: [sin(f0*x), cos(f0*x), sin(f1*x), cos(f1*x), ...]
        # Stack along a new dimension and then flatten
        encoded = torch.stack([sin_features, cos_features], dim=-1)  # (batch, input_dim, num_freq, 2)
        encoded = encoded.reshape(batch_size, input_dim * self.num_frequencies * 2)

        # Optionally prepend original input coordinates
        if self.include_input:
            encoded = torch.cat([coords, encoded], dim=-1)

        return encoded

    @property
    def output_dim(self) -> int:
        """
        Calculate output dimension for a given input dimension.

        Formula:
            - Without input: input_dim × 2 × num_frequencies
            - With input: input_dim × (1 + 2 × num_frequencies)

        For 2D input (x, y) with L=10 frequencies and include_input=True:
            output_dim = 2 × (1 + 2 × 10) = 2 + 40 = 42
        """
        # This will be set properly when forward is called at least once
        # For now, assume 2D input (most common case)
        return self._calculate_output_dim(input_dim=2)

    def _calculate_output_dim(self, input_dim: int) -> int:
        """Calculate output dimension for a specific input dimension."""
        encoding_dim = input_dim * 2 * self.num_frequencies
        if self.include_input:
            encoding_dim += input_dim
        return encoding_dim

    def extra_repr(self) -> str:
        """String representation for printing the module."""
        return (
            f"num_frequencies={self.num_frequencies}, "
            f"include_input={self.include_input}, "
            f"log_sampling={self.log_sampling}, "
            f"scale={self.scale}, "
            f"output_dim={self.output_dim}"
        )


def positional_encode(
    coords: torch.Tensor,
    num_frequencies: int = 10,
    include_input: bool = True,
    log_sampling: bool = True,
    scale: float = 1.0
) -> torch.Tensor:
    """
    Functional interface for positional encoding (stateless).

    This is a convenience function that creates a PositionalEncoding module
    and applies it immediately. Use this for one-off encoding; use the
    PositionalEncoding class for repeated use in a network.

    Args:
        coords: Input coordinates of shape (batch_size, input_dim)
        num_frequencies: Number of frequency bands (L)
        include_input: Whether to include original coordinates
        log_sampling: Use exponential frequency spacing (recommended)
        scale: Frequency scaling factor

    Returns:
        Encoded coordinates of shape (batch_size, output_dim)

    Example:
        >>> coords = torch.tensor([[0.5, -0.3], [0.0, 0.0]])
        >>> encoded = positional_encode(coords, num_frequencies=10)
        >>> print(encoded.shape)  # torch.Size([2, 42])
    """
    encoder = PositionalEncoding(
        num_frequencies=num_frequencies,
        include_input=include_input,
        log_sampling=log_sampling,
        scale=scale
    )
    return encoder(coords)


def get_encoding_dim(
    input_dim: int,
    num_frequencies: int = 10,
    include_input: bool = True
) -> int:
    """
    Calculate the output dimension of positional encoding.

    Useful for determining the input size of subsequent network layers.

    Args:
        input_dim: Dimension of input coordinates (e.g., 2 for 2D, 3 for 3D)
        num_frequencies: Number of frequency bands (L)
        include_input: Whether original coordinates are included

    Returns:
        Output dimension after encoding

    Formula:
        - Without input: input_dim × 2 × num_frequencies
        - With input: input_dim × (1 + 2 × num_frequencies)

    Example:
        >>> dim = get_encoding_dim(input_dim=2, num_frequencies=10, include_input=True)
        >>> print(dim)  # 42
        >>> dim = get_encoding_dim(input_dim=3, num_frequencies=10, include_input=True)
        >>> print(dim)  # 63 (for 3D coordinates like in NeRF)
    """
    encoding_dim = input_dim * 2 * num_frequencies
    if include_input:
        encoding_dim += input_dim
    return encoding_dim
