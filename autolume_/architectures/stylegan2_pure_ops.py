"""
Pure PyTorch implementations of StyleGAN2 custom operations.
These replace the C++/CUDA custom ops to enable MPS compatibility.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ActivationFunc:
    """Activation function descriptor"""
    def_alpha: float = 0
    def_gain: float = 1.0
    cuda_idx: int = 0
    ref: str = ''
    has_2nd_grad: bool = False


# Activation function definitions (compatible with original bias_act)
activation_funcs = {
    'linear':   ActivationFunc(def_alpha=0,    def_gain=1.0,           cuda_idx=1),
    'relu':     ActivationFunc(def_alpha=0,    def_gain=1.41421356237, cuda_idx=2),
    'lrelu':    ActivationFunc(def_alpha=0.2,  def_gain=1.41421356237, cuda_idx=3),
    'tanh':     ActivationFunc(def_alpha=0,    def_gain=1.0,           cuda_idx=4),
    'sigmoid':  ActivationFunc(def_alpha=0,    def_gain=1.0,           cuda_idx=5),
    'elu':      ActivationFunc(def_alpha=1,    def_gain=1.0,           cuda_idx=6),
    'selu':     ActivationFunc(def_alpha=1.67326324, def_gain=1.05070098,  cuda_idx=7),
    'softplus': ActivationFunc(def_alpha=0,    def_gain=1.0,           cuda_idx=8),
    'swish':    ActivationFunc(def_alpha=0,    def_gain=1.0,           cuda_idx=9),
}


def modulated_conv2d(
    x,                  # Input tensor: [batch, in_channels, height, width]
    weight,             # Weight tensor: [out_channels, in_channels, kernel_h, kernel_w]
    styles,             # Style tensor: [batch, in_channels]
    noise=None,         # Optional noise: [batch, 1, height, width]
    up=1,               # Upsampling factor
    down=1,             # Downsampling factor (not typically used)
    padding=0,          # Padding
    resample_filter=None,  # Filter for up/downsampling
    demodulate=True,    # Apply weight demodulation
    flip_weight=True,   # Flip weight for convolution
    fused_modconv=True  # Fused modulation (ignored in pure PyTorch, always fused)
):
    """
    Pure PyTorch implementation of modulated convolution.
    This is the core operation of StyleGAN2 synthesis.
    """
    batch, in_channels, height, width = x.shape
    out_channels, in_channels, kh, kw = weight.shape

    # 1. Modulate weights by style
    # styles: [batch, in_channels] -> [batch, 1, in_channels, 1, 1]
    styles = styles.reshape(batch, 1, in_channels, 1, 1)
    weight = weight.unsqueeze(0)  # [1, out_channels, in_channels, kh, kw]
    weight = weight * styles  # [batch, out_channels, in_channels, kh, kw]

    # 2. Demodulate weights (normalize to preserve variance)
    if demodulate:
        # Compute norm across in_channels and spatial dimensions
        d = torch.rsqrt((weight ** 2).sum(dim=[2, 3, 4], keepdim=True) + 1e-8)
        weight = weight * d

    # 3. Apply upsampling if needed
    if up > 1:
        # Upsample input
        x = F.interpolate(x, scale_factor=up, mode='nearest')
        height *= up
        width *= up

    # 4. Reshape for grouped convolution
    # Treat batch dimension as groups
    x = x.reshape(1, batch * in_channels, height, width)
    weight = weight.reshape(batch * out_channels, in_channels, kh, kw)

    # 5. Apply convolution
    x = F.conv2d(x, weight, padding=padding, groups=batch)

    # 6. Add noise if provided
    if noise is not None:
        x = x + noise

    # 7. Reshape back to [batch, out_channels, height, width]
    _, _, height, width = x.shape
    x = x.reshape(batch, out_channels, height, width)

    return x


def bias_act(
    x,                  # Input tensor
    b=None,             # Bias tensor: [channels]
    dim=1,              # Channel dimension
    act='linear',       # Activation: 'linear', 'lrelu', 'relu', etc.
    alpha=None,         # Negative slope for leaky_relu
    gain=1.0,           # Scaling factor
    clamp=None          # Clamp output to [-clamp, clamp]
):
    """
    Pure PyTorch implementation of bias + activation.
    Fuses bias addition with activation for efficiency.
    """
    # Add bias
    if b is not None:
        if x.ndim == 4:  # [batch, channels, height, width]
            b = b.reshape(1, -1, 1, 1)
        elif x.ndim == 2:  # [batch, channels]
            b = b.reshape(1, -1)
        x = x + b

    # Apply activation
    if act == 'linear':
        pass
    elif act == 'lrelu':
        negative_slope = alpha if alpha is not None else 0.2
        x = F.leaky_relu(x, negative_slope)
    elif act == 'relu':
        x = F.relu(x)
    elif act == 'tanh':
        x = torch.tanh(x)
    elif act == 'sigmoid':
        x = torch.sigmoid(x)
    else:
        raise ValueError(f'Unknown activation: {act}')

    # Apply gain
    if gain != 1.0:
        x = x * gain

    # Apply clamping
    if clamp is not None:
        x = torch.clamp(x, -clamp, clamp)

    return x


def setup_filter(f, device=torch.device('cpu'), normalize=True, flip_filter=False, gain=1, separable=None):
    """
    Setup a 2D FIR filter for upfirdn2d.
    Compatible with original upfirdn2d.setup_filter.
    """
    if f is None:
        return None

    # Convert to tensor if needed
    if not isinstance(f, torch.Tensor):
        f = torch.tensor(f, dtype=torch.float32)

    # Ensure it's 1D or 2D
    assert f.ndim in [1, 2]

    # Make 2D if 1D (outer product)
    if f.ndim == 1:
        f = f.unsqueeze(0) * f.unsqueeze(1)

    # Normalize
    if normalize:
        f = f / f.sum()

    # Apply gain
    if gain != 1:
        f = f * gain

    # Flip if requested
    if flip_filter:
        f = f.flip([0, 1])

    return f.to(device)


def upfirdn2d(x, f, up=1, down=1, padding=0, flip_filter=False, gain=1):
    """
    Pure PyTorch implementation of upfirdn2d (upsample + FIR filter + downsample).
    Simplified version that handles the common cases in StyleGAN2.
    """
    batch, channels, height, width = x.shape

    # Handle filter
    if f is None:
        # No filtering, just up/downsample
        if up > 1:
            x = F.interpolate(x, scale_factor=up, mode='nearest')
        if down > 1:
            x = x[:, :, ::down, ::down]
        return x * gain

    # Ensure filter is 2D
    if f.ndim == 1:
        f = f.unsqueeze(0) * f.unsqueeze(1)  # Outer product for 2D kernel

    if flip_filter:
        f = f.flip([0, 1])

    # Normalize filter
    f = f * (gain / f.sum())

    # Prepare filter for conv2d: [out_ch, in_ch, kh, kw]
    # We want to apply same filter to all channels
    f = f.unsqueeze(0).unsqueeze(0)  # [1, 1, kh, kw]
    f = f.repeat(channels, 1, 1, 1)  # [channels, 1, kh, kw]

    # Upsample
    if up > 1:
        x = F.interpolate(x, scale_factor=up, mode='nearest')

    # Apply filter using depthwise convolution
    x = F.conv2d(x, f, padding=padding, groups=channels)

    # Downsample
    if down > 1:
        x = x[:, :, ::down, ::down]

    return x


def upsample2d(x, f, up=2, padding=0, flip_filter=False, gain=1):
    """Helper function for upsampling with optional filtering"""
    return upfirdn2d(x, f, up=up, down=1, padding=padding, flip_filter=flip_filter, gain=gain)


def downsample2d(x, f, down=2, padding=0, flip_filter=False, gain=1):
    """Helper function for downsampling with optional filtering"""
    return upfirdn2d(x, f, up=1, down=down, padding=padding, flip_filter=flip_filter, gain=gain)


# Wrapper class to maintain compatibility with existing code
class Conv2dGradFix:
    """Dummy wrapper for gradient fixing (not needed in pure PyTorch)"""
    @staticmethod
    def conv2d(*args, **kwargs):
        return F.conv2d(*args, **kwargs)

    @staticmethod
    def conv_transpose2d(*args, **kwargs):
        return F.conv_transpose2d(*args, **kwargs)
