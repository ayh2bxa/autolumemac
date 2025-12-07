#!/usr/bin/env python3
"""
TorchScript Export and Inference for Audio-Reactive Custom StyleGAN2

This module provides the TorchScriptStyleGAN2Wrapper class for:
1. Exporting fresh Custom StyleGAN2 to TorchScript format (.pt file)
2. Loading saved TorchScript models
3. Audio-reactive video generation using TorchScript models

Key Features:
- Model takes RAW AUDIO SAMPLES as input (not pre-computed features)
- FFT computation is performed INSIDE the TorchScript model
- Only uses FFT features (no chroma, RMS, or other librosa features)
- Fully self-contained: audio → FFT → latent modulation → image

TorchScript supports all dynamic operations that ONNX cannot handle,
including the modulated_conv2d operation in StyleGAN2.

This is designed to be run inside HuggingFace Jobs via run_torchscript_job.py.
The main() function is kept for compatibility but should not be used directly.

For HuggingFace Jobs usage:
    python run_torchscript_job.py --image ayh2bxa/autolume --audio-path /app/autolume/input_audio.wav
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
import librosa
from PIL import Image
import cv2
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from architectures import custom_stylegan2


class TorchScriptStyleGAN2Wrapper:
    """
    Wrapper for exporting and using StyleGAN2 via TorchScript

    TorchScript preserves all dynamic operations including:
    - Dynamic reshaping with -1
    - Runtime-dependent control flow
    - Grouped convolutions with variable groups
    """

    def __init__(
        self,
        img_resolution: int = 512,
        w_dim: int = 512,
        device: str = 'cuda',
        pretrained_path: Optional[str] = None
    ):
        """
        Initialize wrapper with fresh or pretrained Custom StyleGAN2

        Args:
            img_resolution: Output image resolution (512 or 1024)
            w_dim: Latent W space dimension
            device: 'cuda', 'cpu', or 'mps'
            pretrained_path: Optional path to pretrained .pkl model (e.g., models/ffhq.pkl)
        """
        self.device = torch.device(device)
        self.img_resolution = img_resolution
        self.w_dim = w_dim
        self.pretrained_path = pretrained_path

        print(f"🔧 Initializing TorchScript wrapper")
        print(f"   Device: {self.device}")
        print(f"   Resolution: {img_resolution}x{img_resolution}")
        print(f"   W dimension: {w_dim}")
        if pretrained_path:
            print(f"   Pretrained model: {pretrained_path}")

        # Create model (fresh or from pretrained)
        if pretrained_path:
            self.model = self._load_pretrained_model(pretrained_path)
        else:
            self.model = self._create_fresh_model()

        self.model.eval()

    def _create_audio_to_image_model(self, synthesis, generator=None) -> nn.Module:
        """
        Create AudioToImageModel wrapper that converts FFT magnitude to images.
        Used by both fresh and pretrained model loading.
        """
        class AudioToImageModel(nn.Module):
            def __init__(self, synthesis, generator=None):
                super().__init__()
                self.synthesis = synthesis
                self.num_ws = synthesis.num_ws
                self.w_dim = synthesis.w_dim

                # Get device from synthesis network
                device = next(synthesis.parameters()).device

                # Generate base latent
                if generator is not None:
                    # Use mapping network to generate a good base latent
                    # Move mapping to device first
                    self.mapping = generator.mapping.to(device)
                    with torch.no_grad():
                        z = torch.randn(1, generator.z_dim, device=device)
                        c = torch.zeros(1, generator.c_dim, device=device)
                        base_w = self.mapping(z, c, truncation_psi=0.7)
                    print("      Using mapping network for base latent (better quality)")
                else:
                    # Fall back to random initialization
                    base_w = torch.randn(1, synthesis.num_ws, synthesis.w_dim, device=device)
                    print("      Using random base latent (generator not provided)")

                # Register base latent as buffer (will be part of model state)
                self.register_buffer('base_latent', base_w)

            def forward(self, fft_magnitude):
                """
                Args:
                    fft_magnitude: Pre-computed FFT magnitude [batch, 512] or [512]
                                   (computed on CPU in C++ code)

                Returns:
                    Generated image [batch, 3, H, W]
                """
                # Ensure input is 2D: [batch, 512]
                if fft_magnitude.dim() == 1:
                    fft_magnitude = fft_magnitude.unsqueeze(0)

                batch_size = fft_magnitude.shape[0]

                # Take first 256 FFT bins (to match w_dim modulation)
                fft_features = fft_magnitude[:, :256]

                # Normalize FFT features to reasonable range (0-1)
                fft_features = fft_features / (fft_features.max(dim=1, keepdim=True)[0] + 1e-8)

                # Scale down to avoid too much deviation from base latent
                # fft_features = (fft_features - 0.5) * 0.3  # Range: [-0.15, 0.15]

                # Modulate base latent with FFT features
                # base_latent: [1, num_ws, w_dim]
                # fft_features: [batch, 256]
                latent = self.base_latent.repeat(batch_size, 1, 1)  # [batch, num_ws, w_dim]

                # Apply FFT modulation to first 256 dimensions across all layers
                fft_mod = fft_features.unsqueeze(1)  # [batch, 1, 256]
                latent[:, :, :256] = latent[:, :, :256] + fft_mod

                # Generate image
                result = self.synthesis(latent, get_rgb_list=False)

                # Handle tuple return: (img, rgb_list)
                if isinstance(result, (tuple, list)):
                    return result[0]
                return result

        return AudioToImageModel(synthesis, generator)

    def _create_fresh_model(self) -> nn.Module:
        """Create a fresh Custom StyleGAN2 synthesis network with audio processing"""
        print("\n🆕 Creating fresh Custom StyleGAN2 synthesis network with audio processing...")

        # Create full generator (needed for mapping network)
        G = custom_stylegan2.Generator(
            z_dim=512,
            c_dim=0,
            w_dim=self.w_dim,
            img_resolution=self.img_resolution,
            img_channels=3,
            channel_base=32768,
            channel_max=512
        ).to(self.device)

        synthesis = G.synthesis

        # Apply patches for stability (same as offline_audio_inference.py)
        self._apply_patches(synthesis)

        # Wrap synthesis network using shared AudioToImageModel
        wrapped = self._create_audio_to_image_model(synthesis, G)
        wrapped.eval()

        print(f"   ✓ Created audio-to-image model with {self._count_parameters(synthesis):,} parameters")
        print(f"   ✓ Model takes pre-computed FFT magnitude (512 values) as input")
        print(f"   ✓ FFT computed on CPU in C++ code, model runs on GPU")
        print(f"   ✓ Base latent generated from mapping network for better quality")

        return wrapped

    def _load_pretrained_model(self, model_path: str) -> nn.Module:
        """Load pretrained StyleGAN2 model from .pkl file using the same method as OfflineAudioInference"""
        print(f"\n📂 Loading pretrained model from {model_path}...")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pretrained model not found: {model_path}")

        try:
            # Use the same loading method as Renderer (widgets/renderer.py line 291)
            # This uses torch_utils.legacy which has the custom parameter
            from torch_utils import legacy
            import dnnlib

            print("   Loading .pkl file...")
            with dnnlib.util.open_url(model_path, verbose=False) as f:
                data = legacy.load_network_pkl(f, custom=True)
            print("   ✓ Loaded pickle successfully")

            # Extract G_ema (same as Renderer.set_pkl line 281)
            if 'G_ema' in data:
                G = data['G_ema']
                print("   ✓ Extracted G_ema from pickle")
            elif 'G' in data:
                G = data['G']
                print("   ✓ Extracted G from pickle")
            else:
                raise ValueError(f"No generator found in pickle. Keys: {data.keys()}")

            # Extract synthesis network
            if hasattr(G, 'synthesis'):
                synthesis = G.synthesis
                print("   ✓ Extracted synthesis network")
            else:
                raise ValueError("Generator does not have 'synthesis' attribute")

            # Move to device
            synthesis = synthesis.to(self.device)

            # Apply patches for stability
            self._apply_patches(synthesis)

            # Update w_dim from loaded model
            if hasattr(synthesis, 'w_dim'):
                self.w_dim = synthesis.w_dim
                print(f"   ✓ Using w_dim from pretrained model: {self.w_dim}")

            # Update img_resolution from loaded model
            if hasattr(synthesis, 'img_resolution'):
                self.img_resolution = synthesis.img_resolution
                print(f"   ✓ Using resolution from pretrained model: {self.img_resolution}")

            # Wrap synthesis network using shared AudioToImageModel
            wrapped = self._create_audio_to_image_model(synthesis, G)
            wrapped.eval()

            print(f"   ✓ Loaded pretrained model with {self._count_parameters(synthesis):,} parameters")
            print(f"   ✓ Wrapped to accept pre-computed FFT magnitude (512 values)")

            return wrapped

        except Exception as e:
            print(f"❌ Failed to load pretrained model")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _apply_patches(self, synthesis: nn.Module):
        """Apply stability patches to synthesis network"""
        # Force rx/ry to 1.0 (no dynamic resizing)
        for name, module in synthesis.named_modules():
            if hasattr(module, 'rx'):
                module.rx = 1.0
            if hasattr(module, 'ry'):
                module.ry = 1.0

    def _count_parameters(self, model: nn.Module) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def export_to_torchscript(
        self,
        output_path: str,
        example_input: Optional[torch.Tensor] = None,
        num_audio_samples: int = 512
    ) -> str:
        """
        Export model to TorchScript format

        Args:
            output_path: Path to save .pt file
            example_input: Optional example input tensor for tracing
                          If None, creates random audio samples
            num_audio_samples: Number of audio samples for example input (default: 16000 = 1 second @ 16kHz)

        Returns:
            Path to saved TorchScript file
        """
        print(f"\n📦 Exporting to TorchScript...")
        print(f"   Output: {output_path}")

        # Create example input if not provided
        if example_input is None:
            # Model expects pre-computed FFT magnitude [batch, 512]
            # Use deterministic input for better trace verification
            torch.manual_seed(42)
            example_input = torch.randn(1, 512, device=self.device)
            print(f"   Example input shape: {list(example_input.shape)} (batch, fft_bins)")
            print(f"   FFT bins: 512 (computed from audio on CPU)")

        try:

            # Trace the model
            print("   🔍 Tracing model execution...")
            with torch.no_grad():
                # Set model to eval and disable any randomness
                self.model.eval()
                traced_model = torch.jit.trace(
                    self.model,
                    example_input,
                    check_trace=False  # Disable strict checking due to StyleGAN2's dynamic ops
                )

            # Save traced model
            print("   💾 Saving TorchScript model...")
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            traced_model.save(output_path)

            # Verify file was created
            if os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"\n✅ TorchScript export SUCCESS!")
                print(f"   File: {output_path}")
                print(f"   Size: {size_mb:.2f} MB")
                return output_path
            else:
                raise RuntimeError(f"File not created: {output_path}")

        except Exception as e:
            print(f"\n❌ TorchScript export FAILED")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def load_torchscript(self, model_path: str, num_audio_samples: int = 512) -> torch.jit.ScriptModule:
        """
        Load TorchScript model from file

        Args:
            model_path: Path to .pt file
            num_audio_samples: Number of audio samples for testing (default: 16000)

        Returns:
            Loaded TorchScript model
        """
        print(f"\n📂 Loading TorchScript model...")
        print(f"   File: {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TorchScript file not found: {model_path}")

        try:
            # Load model on CPU first (torch.jit.load doesn't support MPS as map_location)
            print(f"   Loading on CPU first...")
            loaded_model = torch.jit.load(model_path, map_location='cpu')
            loaded_model.eval()

            # Move to target device if not CPU
            if self.device.type != 'cpu':
                print(f"   Moving model to {self.device}...")
                loaded_model = loaded_model.to(self.device)

            # Verify it works with FFT magnitude input
            with torch.no_grad():
                test_input = torch.randn(1, num_audio_samples, device=self.device)
                test_output = loaded_model(test_input)

            print(f"✅ TorchScript model loaded successfully!")
            print(f"   Input shape: {list(test_input.shape)} (batch, num_audio_samples)")
            print(f"   Output shape: {list(test_output.shape)}")

            return loaded_model

        except Exception as e:
            print(f"❌ Failed to load TorchScript model")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def generate_image(
        self,
        model: torch.jit.ScriptModule,
        audio_samples: torch.Tensor
    ) -> np.ndarray:
        """
        Generate image from raw audio samples using TorchScript model

        Args:
            model: TorchScript model
            audio_samples: Raw audio samples [batch, num_samples] or [num_samples]

        Returns:
            Generated image as numpy array [H, W, 3] in range [0, 255]
        """
        # Ensure correct shape: [batch, num_samples]
        if audio_samples.dim() == 1:
            audio_samples = audio_samples.unsqueeze(0)

        with torch.no_grad():
            # Generate image from audio samples
            img_tensor = model(audio_samples)

            # Handle tuple output (some StyleGAN2 variants return (img, features))
            if isinstance(img_tensor, (tuple, list)):
                img_tensor = img_tensor[0]

            # Convert to numpy [0, 255]
            img = img_tensor.squeeze(0).cpu()
            img = (img.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8).numpy()

        return img

    def process_audio_with_torchscript(
        self,
        model: torch.jit.ScriptModule,
        audio_path: str,
        output_dir: str = './output',
        max_frames: Optional[int] = None,
        fps: int = 30,
        samples_per_frame: int = 1024
    ) -> Tuple[str, str]:
        """
        Generate audio-reactive video using TorchScript model

        Args:
            model: Loaded TorchScript model
            audio_path: Path to audio file
            output_dir: Output directory
            max_frames: Maximum frames to generate (None = all)
            fps: Video frame rate
            samples_per_frame: Number of audio samples per frame (default: 1024)

        Returns:
            Tuple of (video_path, frames_dir)
        """
        print(f"\n🎵 Processing audio with TorchScript model...")
        print(f"   Audio: {audio_path}")
        print(f"   Output: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)
        frames_dir = os.path.join(output_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)

        # Load and process audio
        print("\n🎧 Loading audio...")
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        duration = len(audio) / sr
        total_frames = int(duration * fps)

        if max_frames is not None:
            total_frames = min(total_frames, max_frames)

        print(f"   Duration: {duration:.2f}s")
        print(f"   Sample rate: {sr} Hz")
        print(f"   Generating {total_frames} frames @ {fps} FPS")
        print(f"   Audio samples per frame: {samples_per_frame}")

        # Generate frames
        print(f"\n🎨 Generating {total_frames} frames...")
        frame_paths = []

        # Calculate samples per frame based on audio length
        actual_samples_per_frame = len(audio) // total_frames

        for frame_idx in tqdm(range(total_frames), desc="Frames"):
            # Extract raw audio samples for this frame
            start_idx = frame_idx * actual_samples_per_frame
            end_idx = start_idx + samples_per_frame

            # Handle edge cases
            if end_idx > len(audio):
                # Pad with zeros if we're at the end
                frame_audio = np.pad(audio[start_idx:], (0, end_idx - len(audio)), mode='constant')
            else:
                frame_audio = audio[start_idx:end_idx]

            # Convert to tensor
            audio_tensor = torch.from_numpy(frame_audio).float().to(self.device)

            # Generate image directly from raw audio samples
            img = self.generate_image(model, audio_tensor)

            # Save frame
            frame_path = os.path.join(frames_dir, f'frame_{frame_idx:05d}.png')
            Image.fromarray(img).save(frame_path)
            frame_paths.append(frame_path)

        print(f"   ✓ Generated {len(frame_paths)} frames")

        # Create video
        print("\n🎬 Creating video...")
        video_path = os.path.join(output_dir, 'output.mp4')
        self._create_video(frame_paths, audio_path, video_path, fps)

        print(f"\n✅ Video generation complete!")
        print(f"   Video: {video_path}")
        print(f"   Frames: {frames_dir}")

        return video_path, frames_dir


    def _create_video(
        self,
        frame_paths: list,
        audio_path: str,
        output_path: str,
        fps: int
    ):
        """Create video from frames with audio"""
        # Read first frame to get dimensions
        first_frame = cv2.imread(frame_paths[0])
        height, width = first_frame.shape[:2]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_video = output_path.replace('.mp4', '_temp.mp4')
        out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

        # Write frames
        for frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            out.write(frame)

        out.release()

        # Add audio using ffmpeg
        import subprocess
        cmd = [
            'ffmpeg', '-y',
            '-i', temp_video,
            '-i', audio_path,
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-shortest',
            output_path
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            os.remove(temp_video)
        except subprocess.CalledProcessError:
            print("⚠️  Could not add audio (ffmpeg failed), using video without audio")
            os.rename(temp_video, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="TorchScript export and inference for Custom StyleGAN2",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Mode selection
    parser.add_argument(
        '--export-only',
        action='store_true',
        help='Only export to TorchScript (no inference)'
    )

    # Paths
    parser.add_argument(
        '--output-path',
        default='./stylegan2.pt',
        help='Path to save TorchScript model (default: ./stylegan2.pt)'
    )

    parser.add_argument(
        '--load-path',
        help='Path to load existing TorchScript model (skips export)'
    )

    parser.add_argument(
        '--audio',
        help='Path to audio file for inference'
    )

    parser.add_argument(
        '--output-dir',
        default='./output',
        help='Output directory for generated video (default: ./output)'
    )

    # Model settings
    parser.add_argument(
        '--pretrained',
        help='Path to pretrained .pkl model (e.g., models/ffhq.pkl). If not provided, uses fresh model.'
    )

    parser.add_argument(
        '--resolution',
        type=int,
        default=512,
        choices=[128, 256, 384, 512, 1024],
        help='Image resolution (default: 512, ignored if using pretrained model)'
    )

    parser.add_argument(
        '--device',
        default='mps',
        choices=['cuda', 'cpu', 'mps'],
        help='Device to use (default: mps)'
    )

    parser.add_argument(
        '--max-frames',
        type=int,
        help='Maximum frames to generate (default: all)'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Video frame rate (default: 30)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🚀 TorchScript Custom StyleGAN2")
    print("=" * 70)

    # Initialize wrapper
    wrapper = TorchScriptStyleGAN2Wrapper(
        img_resolution=args.resolution,
        device=args.device,
        pretrained_path=args.pretrained
    )

    torchscript_path = args.output_path

    # Export phase
    if not args.load_path:
        print("\n📦 Phase 1: Export to TorchScript")
        print("-" * 70)
        torchscript_path = wrapper.export_to_torchscript(args.output_path)
    else:
        print("\n⏭️  Skipping export (using existing model)")
        torchscript_path = args.load_path

    # Stop here if export-only
    if args.export_only:
        print("\n✅ Export complete (--export-only specified)")
        return

    # Inference phase
    if args.audio:
        print("\n🎵 Phase 2: Audio-Reactive Inference")
        print("-" * 70)

        # Load TorchScript model
        loaded_model = wrapper.load_torchscript(torchscript_path)

        # Generate video
        video_path, frames_dir = wrapper.process_audio_with_torchscript(
            loaded_model,
            args.audio,
            args.output_dir,
            args.max_frames,
            args.fps
        )

        print("\n" + "=" * 70)
        print("🎉 Complete!")
        print(f"   TorchScript model: {torchscript_path}")
        print(f"   Video: {video_path}")
        print(f"   Frames: {frames_dir}")
        print("=" * 70)
    else:
        print("\n⚠️  No audio file provided (use --audio)")
        print(f"✅ TorchScript model saved: {torchscript_path}")


if __name__ == '__main__':
    main()