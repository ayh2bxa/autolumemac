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


def _conv2d_resample_pure(x, w, f=None, up=1, down=1, padding=0, groups=1, flip_weight=True):
    """
    Pure PyTorch implementation of conv2d_resample.
    Handles convolution with up/downsampling and filtering.
    """
    batch, in_channels, height, width = x.shape
    out_channels, in_channels_per_group, kh, kw = w.shape

    # Parse padding
    if isinstance(padding, int):
        px0 = px1 = py0 = py1 = padding
    elif len(padding) == 2:
        py0 = py1 = padding[0]
        px0 = px1 = padding[1]
    elif len(padding) == 4:
        px0, px1, py0, py1 = padding
    else:
        px0 = px1 = py0 = py1 = 0

    # Get filter size
    if f is None:
        fw = fh = 1
    elif f.ndim == 1:
        fw = fh = f.shape[0]
    else:
        fh, fw = f.shape

    # Adjust padding for up/downsampling (like conv2d_resample.py lines 82-91)
    if up > 1:
        px0 += (fw + up - 1) // 2
        px1 += (fw - up) // 2
        py0 += (fh + up - 1) // 2
        py1 += (fh - up) // 2
    if down > 1:
        px0 += (fw - down + 1) // 2
        px1 += (fw - down) // 2
        py0 += (fh - down + 1) // 2
        py1 += (fh - down) // 2

    # Downsampling only (conv2d_resample.py lines 106-109)
    if down > 1 and up == 1:
        x = upfirdn2d(x, f, padding=[px0, px1, py0, py1], flip_filter=False)
        x = F.conv2d(x, w, stride=down, groups=groups)
        return x

    # Upsampling (conv2d_resample.py lines 112-129)
    if up > 1:
        # Transpose weights for transposed convolution
        if groups == 1:
            w = w.transpose(0, 1)
        else:
            w = w.reshape(groups, out_channels // groups, in_channels_per_group, kh, kw)
            w = w.transpose(1, 2)
            w = w.reshape(groups * in_channels_per_group, out_channels // groups, kh, kw)

        # Adjust padding for transposed conv
        px0 -= kw - 1
        px1 -= kw - up
        py0 -= kh - 1
        py1 -= kh - up
        pxt = max(min(-px0, -px1), 0)
        pyt = max(min(-py0, -py1), 0)

        # Transposed convolution
        x = F.conv_transpose2d(x, w, stride=up, padding=[pyt, pxt], groups=groups)

        # Apply filter
        x = upfirdn2d(x, f, padding=[px0 + pxt, px1 + pxt, py0 + pyt, py1 + pyt],
                     gain=up ** 2, flip_filter=False)

        # Downsample if needed
        if down > 1:
            x = upfirdn2d(x, f, down=down, flip_filter=False)
        return x

    # No resampling (conv2d_resample.py lines 132-134)
    if up == 1 and down == 1:
        if px0 == px1 and py0 == py1 and px0 >= 0 and py0 >= 0:
            return F.conv2d(x, w, padding=[py0, px0], groups=groups)

    # Fallback: shouldn't reach here for StyleGAN2
    x = F.conv2d(x, w, padding=padding, groups=groups)
    return x


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
    fused_modconv=True  # Fused modulation (modulate weights vs activations)
):
    """
    Pure PyTorch implementation of modulated convolution.
    This is the core operation of StyleGAN2 synthesis.
    Implements both fused and non-fused paths to match CUDA version.
    """
    batch, in_channels, height, width = x.shape
    out_channels, in_channels_w, kh, kw = weight.shape

    # Calculate per-sample weights and demodulation coefficients
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0)  # [1, out_channels, in_channels, kh, kw]
        w = w * styles.reshape(batch, 1, -1, 1, 1)  # [batch, out_channels, in_channels, kh, kw]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [batch, out_channels]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch, -1, 1, 1, 1)  # Apply demod to weights

    # Execute by scaling the activations before and after convolution (NON-FUSED)
    if not fused_modconv:
        x = x * styles.reshape(batch, -1, 1, 1)  # Modulate activations
        x = _conv2d_resample_pure(x, weight, f=resample_filter, up=up, down=down,
                                  padding=padding, groups=1, flip_weight=flip_weight)
        if demodulate and noise is not None:
            # Apply demod and add noise (fma: x * a + b)
            x = x * dcoefs.reshape(batch, -1, 1, 1) + noise
        elif demodulate:
            x = x * dcoefs.reshape(batch, -1, 1, 1)
        elif noise is not None:
            x = x + noise
        return x

    # Execute as one fused op using grouped convolution (FUSED)
    x = x.reshape(1, -1, height, width)  # [1, batch*in_channels, height, width]
    w = w.reshape(-1, in_channels_w, kh, kw)  # [batch*out_channels, in_channels, kh, kw]
    x = _conv2d_resample_pure(x, w, f=resample_filter, up=up, down=down,
                              padding=padding, groups=batch, flip_weight=flip_weight)
    x = x.reshape(batch, -1, x.shape[2], x.shape[3])  # [batch, out_channels, height, width]
    if noise is not None:
        x = x + noise
    return x


def bias_act(
    x,                  # Input tensor
    b=None,             # Bias tensor: [channels]
    dim=1,              # Channel dimension
    act='linear',       # Activation: 'linear', 'lrelu', 'relu', etc.
    alpha=None,         # Negative slope for leaky_relu
    gain=None,          # Scaling factor (None = use activation's default)
    clamp=None          # Clamp output to [-clamp, clamp]
):
    """
    Pure PyTorch implementation of bias + activation.
    Fuses bias addition with activation for efficiency.
    Compatible with torch_utils.ops.bias_act.
    """
    # Get activation spec and default parameters
    assert act in activation_funcs, f'Unknown activation: {act}'
    spec = activation_funcs[act]
    alpha = float(alpha if alpha is not None else spec.def_alpha)
    gain = float(gain if gain is not None else spec.def_gain)
    clamp = float(clamp if clamp is not None else -1)

    # Add bias
    if b is not None:
        assert b.ndim == 1
        assert 0 <= dim < x.ndim
        assert b.shape[0] == x.shape[dim]
        x = x + b.reshape([-1 if i == dim else 1 for i in range(x.ndim)])

    # Apply activation with alpha parameter
    if act == 'linear':
        pass
    elif act == 'lrelu':
        x = F.leaky_relu(x, alpha)
    elif act == 'relu':
        x = F.relu(x)
    elif act == 'tanh':
        x = torch.tanh(x)
    elif act == 'sigmoid':
        x = torch.sigmoid(x)
    elif act == 'elu':
        x = F.elu(x, alpha)
    elif act == 'selu':
        x = F.selu(x)
    elif act == 'softplus':
        x = F.softplus(x)
    elif act == 'swish':
        x = torch.sigmoid(x) * x
    else:
        raise ValueError(f'Unknown activation: {act}')

    # Apply gain
    if gain != 1.0:
        x = x * gain

    # Apply clamping
    if clamp >= 0:
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
    Compatible with torch_utils.ops.upfirdn2d
    """
    batch, channels, height, width = x.shape

    # Parse up/down factors
    if isinstance(up, int):
        upx = upy = up
    else:
        upx, upy = up

    if isinstance(down, int):
        downx = downy = down
    else:
        downx, downy = down

    # Handle filter
    if f is None:
        # No filtering, just up/downsample
        if upx > 1 or upy > 1:
            x = F.interpolate(x, scale_factor=(upy, upx), mode='nearest')
        if downx > 1 or downy > 1:
            x = x[:, :, ::downy, ::downx]
        return x * gain

    # Ensure filter is 2D
    if f.ndim == 1:
        f = f.unsqueeze(0) * f.unsqueeze(1)  # Outer product for 2D kernel

    # Setup filter (apply gain before flipping, like reference)
    f = f * (gain ** (f.ndim / 2))
    f = f.to(x.dtype).to(x.device)

    if not flip_filter:
        # Note: reference flips when flip_filter=False (line 207)
        f = f.flip([0, 1])

    # Prepare filter for conv2d: [out_ch, in_ch, kh, kw]
    f = f.unsqueeze(0).unsqueeze(0)  # [1, 1, kh, kw]
    f = f.repeat(channels, 1, 1, 1)  # [channels, 1, kh, kw]

    # Upsample by inserting zeros (not nearest neighbor!)
    if upx > 1 or upy > 1:
        # Reshape to insert dimensions for zero-padding
        x = x.reshape([batch, channels, height, 1, width, 1])
        # Pad with zeros between pixels
        x = F.pad(x, [0, upx - 1, 0, 0, 0, upy - 1])
        # Reshape back to 4D
        x = x.reshape([batch, channels, height * upy, width * upx])

    # Parse padding (like reference implementation)
    if isinstance(padding, int):
        padx0 = padx1 = pady0 = pady1 = padding
    elif len(padding) == 2:
        padx0 = padx1 = padding[0]
        pady0 = pady1 = padding[1]
    elif len(padding) == 4:
        padx0, padx1, pady0, pady1 = padding
    else:
        padx0 = padx1 = pady0 = pady1 = 0

    # Pad or crop (like reference lines 200-201)
    x = F.pad(x, [max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0)])
    x = x[:, :,
          max(-pady0, 0) : x.shape[2] - max(-pady1, 0),
          max(-padx0, 0) : x.shape[3] - max(-padx1, 0)]

    # Apply filter using depthwise convolution
    x = F.conv2d(x, f, padding=0, groups=channels)

    # Downsample
    if downx > 1 or downy > 1:
        x = x[:, :, ::downy, ::downx]

    return x


def upsample2d(x, f, up=2, padding=0, flip_filter=False, gain=1):
    """
    Upsample a batch of 2D images using the given 2D FIR filter.
    Compatible with torch_utils.ops.upfirdn2d.upsample2d
    """
    # Parse up factor
    if isinstance(up, int):
        upx = upy = up
    else:
        upx, upy = up

    # Parse padding
    if isinstance(padding, int):
        padx0 = padx1 = pady0 = pady1 = padding
    elif len(padding) == 2:
        padx0 = padx1 = padding[0]
        pady0 = pady1 = padding[1]
    else:
        padx0, padx1, pady0, pady1 = padding

    # Get filter size
    if f is None:
        fw = fh = 1
    elif f.ndim == 1:
        fw = fh = f.shape[0]
    else:
        fh, fw = f.shape

    # Calculate padding for upsampling (matches CUDA version)
    p = [
        padx0 + (fw + upx - 1) // 2,
        padx1 + (fw - upx) // 2,
        pady0 + (fh + upy - 1) // 2,
        pady1 + (fh - upy) // 2,
    ]

    return upfirdn2d(x, f, up=up, down=1, padding=p, flip_filter=flip_filter, gain=gain * upx * upy)


def downsample2d(x, f, down=2, padding=0, flip_filter=False, gain=1):
    """
    Downsample a batch of 2D images using the given 2D FIR filter.
    Compatible with torch_utils.ops.upfirdn2d.downsample2d
    """
    # Parse down factor
    if isinstance(down, int):
        downx = downy = down
    else:
        downx, downy = down

    # Parse padding
    if isinstance(padding, int):
        padx0 = padx1 = pady0 = pady1 = padding
    elif len(padding) == 2:
        padx0 = padx1 = padding[0]
        pady0 = pady1 = padding[1]
    else:
        padx0, padx1, pady0, pady1 = padding

    # Get filter size
    if f is None:
        fw = fh = 1
    elif f.ndim == 1:
        fw = fh = f.shape[0]
    else:
        fh, fw = f.shape

    # Calculate padding for downsampling (matches CUDA version)
    p = [
        padx0 + (fw - downx + 1) // 2,
        padx1 + (fw - downx) // 2,
        pady0 + (fh - downy + 1) // 2,
        pady1 + (fh - downy) // 2,
    ]

    return upfirdn2d(x, f, up=1, down=down, padding=p, flip_filter=flip_filter, gain=gain)


# Wrapper class to maintain compatibility with existing code
class Conv2dGradFix:
    """Dummy wrapper for gradient fixing (not needed in pure PyTorch)"""
    @staticmethod
    def conv2d(*args, **kwargs):
        return F.conv2d(*args, **kwargs)

    @staticmethod
    def conv_transpose2d(*args, **kwargs):
        return F.conv_transpose2d(*args, **kwargs)
