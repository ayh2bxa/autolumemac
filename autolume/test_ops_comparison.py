#!/usr/bin/env python3
"""
Test script to compare CUDA ops vs Pure PyTorch ops
Identifies numerical differences that could cause inference issues
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Import CUDA ops from custom_stylegan2_win
try:
    from torch_utils.ops import conv2d_resample, bias_act, fma, upfirdn2d
    CUDA_OPS_AVAILABLE = True
    print("✓ CUDA ops imported successfully")
except Exception as e:
    print(f"✗ Failed to import CUDA ops: {e}")
    CUDA_OPS_AVAILABLE = False

# Import pure PyTorch ops
try:
    from architectures import stylegan2_pure_ops
    PURE_OPS_AVAILABLE = True
    print("✓ Pure PyTorch ops imported successfully")
except Exception as e:
    print(f"✗ Failed to import pure ops: {e}")
    PURE_OPS_AVAILABLE = False
    sys.exit(1)


class OpsTester:
    """Test suite for comparing CUDA vs Pure PyTorch operations"""

    def __init__(self, device='cpu', tolerance=1e-5):
        """
        Args:
            device: 'cuda', 'cpu', or 'mps'
            tolerance: Maximum allowed difference for tests to pass
        """
        self.device = torch.device(device)
        self.tolerance = tolerance
        self.results = []

    def log_result(self, test_name, passed, max_diff, mean_diff, details=""):
        """Log test result"""
        result = {
            'test': test_name,
            'passed': passed,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'details': details
        }
        self.results.append(result)

        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"\n{status}: {test_name}")
        print(f"  Max diff: {max_diff:.2e}")
        print(f"  Mean diff: {mean_diff:.2e}")
        if details:
            print(f"  Details: {details}")

    def compare_tensors(self, tensor_cuda, tensor_pure, test_name):
        """Compare two tensors and log results"""
        if tensor_cuda.shape != tensor_pure.shape:
            self.log_result(test_name, False, float('inf'), float('inf'),
                          f"Shape mismatch: {tensor_cuda.shape} vs {tensor_pure.shape}")
            return False

        diff = (tensor_cuda - tensor_pure).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        passed = max_diff < self.tolerance
        self.log_result(test_name, passed, max_diff, mean_diff)
        return passed

    # ===================================================================
    # Test 1: bias_act
    # ===================================================================

    def test_bias_act(self):
        """Test bias_act operation"""
        print("\n" + "="*70)
        print("TEST 1: bias_act")
        print("="*70)

        if not CUDA_OPS_AVAILABLE:
            print("Skipping - CUDA ops not available")
            return

        # Test parameters
        batch_size, channels, height, width = 2, 64, 32, 32

        # Create test inputs
        x = torch.randn(batch_size, channels, height, width, device=self.device)
        b = torch.randn(channels, device=self.device)

        # Test different activations
        for act in ['linear', 'lrelu', 'relu']:
            print(f"\n  Testing activation: {act}")

            # CUDA version
            out_cuda = bias_act.bias_act(x, b, act=act)

            # Pure version
            out_pure = stylegan2_pure_ops.bias_act(x, b, act=act)

            self.compare_tensors(out_cuda, out_pure, f"bias_act_{act}")

    # ===================================================================
    # Test 2: setup_filter
    # ===================================================================

    def test_setup_filter(self):
        """Test filter setup"""
        print("\n" + "="*70)
        print("TEST 2: setup_filter")
        print("="*70)

        if not CUDA_OPS_AVAILABLE:
            print("Skipping - CUDA ops not available")
            return

        # Test common filter configurations
        filters = [
            [1, 3, 3, 1],
            [1, 2, 1],
            None
        ]

        for f in filters:
            print(f"\n  Testing filter: {f}")

            # CUDA version
            filter_cuda = upfirdn2d.setup_filter(f)

            # Pure version
            filter_pure = stylegan2_pure_ops.setup_filter(f)

            if filter_cuda is None and filter_pure is None:
                print("  Both returned None - OK")
                self.log_result(f"setup_filter_{f}", True, 0.0, 0.0, "Both None")
                continue
            elif filter_cuda is None or filter_pure is None:
                self.log_result(f"setup_filter_{f}", False, float('inf'), float('inf'),
                              f"One is None: cuda={filter_cuda is None}, pure={filter_pure is None}")
                continue

            self.compare_tensors(filter_cuda, filter_pure, f"setup_filter_{f}")

    # ===================================================================
    # Test 3: upsample2d
    # ===================================================================

    def test_upsample2d(self):
        """Test 2D upsampling"""
        print("\n" + "="*70)
        print("TEST 3: upsample2d")
        print("="*70)

        if not CUDA_OPS_AVAILABLE:
            print("Skipping - CUDA ops not available")
            return

        # Test inputs
        batch_size, channels, height, width = 2, 64, 16, 16
        x = torch.randn(batch_size, channels, height, width, device=self.device)
        f = upfirdn2d.setup_filter([1, 3, 3, 1])

        # CUDA version
        out_cuda = upfirdn2d.upsample2d(x, f)

        # Pure version
        out_pure = stylegan2_pure_ops.upsample2d(x, f)

        self.compare_tensors(out_cuda, out_pure, "upsample2d")

    # ===================================================================
    # Test 4: downsample2d
    # ===================================================================

    def test_downsample2d(self):
        """Test 2D downsampling"""
        print("\n" + "="*70)
        print("TEST 4: downsample2d")
        print("="*70)

        if not CUDA_OPS_AVAILABLE:
            print("Skipping - CUDA ops not available")
            return

        # Test inputs
        batch_size, channels, height, width = 2, 64, 32, 32
        x = torch.randn(batch_size, channels, height, width, device=self.device)
        f = upfirdn2d.setup_filter([1, 3, 3, 1])

        # CUDA version
        out_cuda = upfirdn2d.downsample2d(x, f)

        # Pure version
        out_pure = stylegan2_pure_ops.downsample2d(x, f)

        self.compare_tensors(out_cuda, out_pure, "downsample2d")

    # ===================================================================
    # Test 5: modulated_conv2d
    # ===================================================================

    def test_modulated_conv2d(self):
        """Test modulated convolution - THE MOST CRITICAL OPERATION"""
        print("\n" + "="*70)
        print("TEST 5: modulated_conv2d (MOST CRITICAL)")
        print("="*70)

        if not CUDA_OPS_AVAILABLE:
            print("Skipping - CUDA ops not available")
            return

        # Test parameters
        batch_size = 2
        in_channels = 512
        out_channels = 512
        height, width = 16, 16
        kernel_size = 3

        # Create test inputs
        x = torch.randn(batch_size, in_channels, height, width, device=self.device)
        weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device=self.device)
        styles = torch.randn(batch_size, in_channels, device=self.device)
        noise = torch.randn(batch_size, out_channels, height, width, device=self.device)

        # Test configurations
        configs = [
            {"demodulate": True, "fused_modconv": True, "up": 1, "down": 1},
            {"demodulate": True, "fused_modconv": False, "up": 1, "down": 1},
            {"demodulate": False, "fused_modconv": True, "up": 1, "down": 1},
            {"demodulate": True, "fused_modconv": True, "up": 2, "down": 1},
            {"demodulate": True, "fused_modconv": True, "up": 1, "down": 2},
        ]

        for i, config in enumerate(configs):
            print(f"\n  Config {i+1}: {config}")

            # Adjust input size for up/down sampling
            h_adj = height * config['down'] // config['up']
            w_adj = width * config['down'] // config['up']
            x_test = torch.randn(batch_size, in_channels, h_adj, w_adj, device=self.device)

            try:
                # CUDA version - using the implementation from custom_stylegan2_win.py
                out_cuda = self._modulated_conv2d_cuda(
                    x_test, weight, styles, noise=None,
                    padding=kernel_size//2,
                    resample_filter=None,
                    **config
                )

                # Pure version
                out_pure = stylegan2_pure_ops.modulated_conv2d(
                    x_test, weight, styles, noise=None,
                    padding=kernel_size//2,
                    resample_filter=None,
                    **config
                )

                self.compare_tensors(out_cuda, out_pure, f"modulated_conv2d_config_{i+1}")
            except Exception as e:
                print(f"  Error: {e}")
                self.log_result(f"modulated_conv2d_config_{i+1}", False,
                              float('inf'), float('inf'), str(e))

    def _modulated_conv2d_cuda(self, x, weight, styles, noise=None, up=1, down=1,
                               padding=0, resample_filter=None, demodulate=True,
                               flip_weight=True, fused_modconv=True):
        """CUDA implementation of modulated_conv2d from custom_stylegan2_win.py"""
        from torch_utils import misc

        batch_size = x.shape[0]
        out_channels, in_channels, kh, kw = weight.shape

        # Pre-normalize inputs to avoid FP16 overflow.
        if x.dtype == torch.float16 and demodulate:
            weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float('inf'), dim=[1,2,3], keepdim=True))
            styles = styles / styles.norm(float('inf'), dim=1, keepdim=True)

        # Calculate per-sample weights and demodulation coefficients.
        w = None
        dcoefs = None
        if demodulate or fused_modconv:
            w = weight.unsqueeze(0)
            w = w * styles.reshape(batch_size, 1, -1, 1, 1)
        if demodulate:
            dcoefs = (w.square().sum(dim=[2,3,4]) + 1e-8).rsqrt()
        if demodulate and fused_modconv:
            w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1)

        # Execute by scaling the activations before and after the convolution.
        if not fused_modconv:
            x = x * styles.to(x.dtype).reshape(batch_size, -1, 1, 1)
            x = conv2d_resample.conv2d_resample(x=x, w=weight.to(x.dtype), f=resample_filter,
                                               up=up, down=down, padding=padding, flip_weight=flip_weight)
            if demodulate and noise is not None:
                x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
            elif demodulate:
                x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
            elif noise is not None:
                x = x.add_(noise.to(x.dtype))
            return x

        # Execute as one fused op using grouped convolution.
        with misc.suppress_tracer_warnings():
            batch_size = int(batch_size)
        x = x.reshape(1, -1, *x.shape[2:])
        w = w.reshape(-1, in_channels, kh, kw)
        x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=resample_filter,
                                           up=up, down=down, padding=padding,
                                           groups=batch_size, flip_weight=flip_weight)
        x = x.reshape(batch_size, -1, *x.shape[2:])
        if noise is not None:
            x = x.add_(noise)
        return x

    # ===================================================================
    # Test 6: Full synthesis pass
    # ===================================================================

    def test_full_synthesis_pass(self):
        """Test full synthesis network pass with pretrained model"""
        print("\n" + "="*70)
        print("TEST 6: Full Synthesis Pass (with pretrained model)")
        print("="*70)

        try:
            import dnnlib
            from torch_utils import legacy

            # Try to load FFHQ model
            model_path = './models/ffhq.pkl'
            if not Path(model_path).exists():
                print(f"  Model not found: {model_path}")
                print("  Skipping full synthesis test")
                return

            print(f"  Loading pretrained model: {model_path}")
            with dnnlib.util.open_url(model_path, verbose=False) as f:
                data = legacy.load_network_pkl(f, custom=True)

            G = data['G_ema']
            synthesis = G.synthesis.to(self.device).eval()

            # Generate test latent
            z = torch.randn(1, G.z_dim, device=self.device)
            c = torch.zeros(1, G.c_dim, device=self.device)

            with torch.no_grad():
                w = G.mapping(z, c, truncation_psi=0.7)

                # Run synthesis
                img, _ = synthesis(w, get_rgb_list=False)

            print(f"  Output shape: {img.shape}")
            print(f"  Output range: [{img.min().item():.3f}, {img.max().item():.3f}]")
            print(f"  Output mean: {img.mean().item():.3f}")
            print(f"  Output std: {img.std().item():.3f}")

            # Check for NaN or Inf
            has_nan = torch.isnan(img).any().item()
            has_inf = torch.isinf(img).any().item()

            if has_nan:
                print("  ✗ WARNING: Output contains NaN values!")
            if has_inf:
                print("  ✗ WARNING: Output contains Inf values!")

            if not has_nan and not has_inf and img.abs().max() < 100:
                print("  ✓ Output appears valid (no NaN/Inf, reasonable range)")
            else:
                print("  ✗ Output may be corrupted")

        except Exception as e:
            print(f"  Error during full synthesis test: {e}")
            import traceback
            traceback.print_exc()

    # ===================================================================
    # Run all tests
    # ===================================================================

    def run_all_tests(self):
        """Run all comparison tests"""
        print("\n" + "="*70)
        print("STYLEGAN2 OPS COMPARISON TEST SUITE")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Tolerance: {self.tolerance}")
        print(f"CUDA ops available: {CUDA_OPS_AVAILABLE}")
        print(f"Pure ops available: {PURE_OPS_AVAILABLE}")

        if not CUDA_OPS_AVAILABLE:
            print("\n⚠️  WARNING: CUDA ops not available. Only testing pure ops validity.")

        # Run tests
        self.test_bias_act()
        self.test_setup_filter()
        self.test_upsample2d()
        self.test_downsample2d()
        self.test_modulated_conv2d()
        self.test_full_synthesis_pass()

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)

        passed = sum(1 for r in self.results if r['passed'])
        failed = len(self.results) - passed

        print(f"Total tests: {len(self.results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")

        if failed > 0:
            print("\n❌ FAILED TESTS:")
            for r in self.results:
                if not r['passed']:
                    print(f"  - {r['test']}: max_diff={r['max_diff']:.2e}")
        else:
            print("\n✅ ALL TESTS PASSED!")

        print("="*70)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Compare CUDA vs Pure PyTorch ops')
    parser.add_argument('--device', default='cpu', choices=['cuda', 'cpu', 'mps'],
                       help='Device to run tests on (default: cpu)')
    parser.add_argument('--tolerance', type=float, default=1e-5,
                       help='Maximum allowed difference (default: 1e-5)')

    args = parser.parse_args()

    # Run tests
    tester = OpsTester(device=args.device, tolerance=args.tolerance)
    tester.run_all_tests()


if __name__ == '__main__':
    main()
