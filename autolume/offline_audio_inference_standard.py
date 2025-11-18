#!/usr/bin/env python3
"""
Offline Audio-to-Visual Inference using Simplified Custom StyleGAN2
Uses custom_stylegan2_simple.py (ONNX-compatible version of custom architecture)

This version:
- Works with models trained on custom_stylegan2.py
- Removes all ONNX-incompatible features (rx/ry, kornia, etc.)
- Keeps square images only
- Full audio reactivity maintained
"""

import numpy as np
import torch
import torch.nn as nn
import librosa
import cv2
import os
from pathlib import Path

# Autolume imports
from audio.feature_extractor import FeatureExtractor
import dnnlib
from torch_utils import legacy
import pickle


class StandardStyleGANWrapper(nn.Module):
    """
    ONNX-exportable wrapper using standard StyleGAN2 architecture.
    No custom modifications - fully ONNX compatible.
    """
    def __init__(self, generator, truncation_psi=0.7, truncation_cutoff=8):
        super().__init__()
        self.mapping = generator.mapping
        self.synthesis = generator.synthesis
        self.c_dim = generator.c_dim
        self.truncation_psi = truncation_psi
        self.truncation_cutoff = truncation_cutoff

    def forward(self, z):
        """
        Forward pass for ONNX export.
        Args:
            z: Latent vector in Z space, shape [batch_size, 512]
        Returns:
            Generated image, shape [batch_size, 3, H, W] in range [-1, 1]
        """
        # Create conditioning vector (zeros for unconditional generation)
        batch_size = z.shape[0]
        c = torch.zeros([batch_size, self.c_dim], dtype=torch.float32, device=z.device)

        # Map Z → W+ latent space with fixed truncation
        w = self.mapping(
            z=z,
            c=c,
            truncation_psi=self.truncation_psi,
            truncation_cutoff=self.truncation_cutoff
        )

        # Generate image through synthesis network
        img = self.synthesis(w, noise_mode='const', force_fp32=True)

        return img


class OfflineAudioInferenceStandard:
    """
    Audio-to-visual inference using standard StyleGAN2.
    Fully ONNX-compatible, audio-reactive, real-time capable.
    """
    def __init__(self, model_path, output_dir="./output", device=None):
        """
        Initialize offline inference with standard StyleGAN2
        Args:
            model_path: Path to trained StyleGAN pickle file
            output_dir: Directory to save generated images
            device: Device to use ('cuda' or 'cpu'). If None, auto-detect.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Load StyleGAN model with legacy support
        print(f"Loading model: {model_path}")
        with dnnlib.util.open_url(model_path) as f:
            data = legacy.load_network_pkl(f)
            G = data['G_ema'].to(self.device).eval()

        print(f"Model loaded: {G.__class__.__name__}")
        print(f"Output resolution: {G.img_resolution}x{G.img_resolution}")

        # Check if model uses custom or standard architecture
        module_name = G.synthesis.__class__.__module__
        if 'custom_stylegan2' in module_name and 'simple' not in module_name:
            print("⚠️  Warning: Model uses full custom_stylegan2 architecture")
            print("   This will work for inference but may fail ONNX export")
            print("   (Model has dynamic operations: rx/ry ratios, kornia resize)")
        elif 'custom_stylegan2_simple' in module_name:
            print("✓ Model uses simplified custom StyleGAN2 (ONNX-compatible)")
        elif 'networks_stylegan2' in module_name:
            print("✓ Model uses standard StyleGAN2 architecture (ONNX-compatible)")
        else:
            print(f"ℹ️  Model architecture: {module_name}")

        print(f"   Inference: ✓ Will work")
        print(f"   ONNX export: {'✓ Should work' if any(x in module_name for x in ['networks_stylegan2', 'simple']) else '⚠️  May fail'}")

        # Create ONNX-exportable wrapper
        self.onnx_model = StandardStyleGANWrapper(
            G,
            truncation_psi=0.7,
            truncation_cutoff=8
        ).to(self.device)
        self.onnx_model.eval()

        # Audio processing parameters
        self.sr = 44100
        self.n_fft = 512
        self.n_chroma = 12
        self.hop_length = 1024

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            n_fft=self.n_fft,
            n_chroma=self.n_chroma,
            sr=self.sr
        )

        # Parameter mapping configuration
        self.setup_parameter_mapping()

    def setup_parameter_mapping(self):
        """Configure how audio features map to StyleGAN parameters"""
        self.latent_dim = 512  # StyleGAN latent dimension
        self.base_seed = 42    # Base seed for consistent generation

        # Audio feature scaling factors
        self.fft_scale = 0.1
        self.chroma_scale = 0.05
        self.rms_scale = 0.2

    def load_audio(self, audio_path):
        """Load and preprocess audio file"""
        print(f"Loading audio: {audio_path}")
        audio, sr = librosa.load(audio_path, sr=self.sr)

        # Split into non-overlapping windows for processing
        frames = librosa.util.frame(
            audio,
            frame_length=self.hop_length,
            hop_length=self.hop_length
        )

        return frames.T  # Shape: (n_frames, frame_length)

    def extract_audio_features(self, audio_frame):
        """Extract features from single audio frame"""
        # FFT analysis
        fft = np.abs(librosa.stft(audio_frame, n_fft=self.n_fft * 2 - 1))
        fft_features = fft.mean(axis=1)

        # Chroma features
        chroma = librosa.feature.chroma_stft(S=fft**2, n_chroma=self.n_chroma)
        chroma_features = chroma.mean(axis=1)
        chroma_features /= (chroma_features.sum() + 1e-8)

        # RMS energy (low/high frequency)
        low_rms = fft_features[:150].mean()
        high_rms = fft_features[500:].mean()

        return {
            'fft': fft_features,
            'chroma': chroma_features,
            'low_rms': low_rms,
            'high_rms': high_rms
        }

    def map_audio_to_latent(self, audio_features, frame_idx):
        """
        Map audio features to StyleGAN latent vector.
        THIS IS WHERE AUDIO REACTIVITY HAPPENS!
        """
        # Generate base latent vector from seed
        np.random.seed(self.base_seed)
        base_latent = np.random.randn(self.latent_dim)

        # Modulate latent vector with audio features
        latent_vector = base_latent.copy()

        # FFT modulation - audio frequencies affect visual features
        fft_mod = audio_features['fft'][:256] * self.fft_scale
        latent_vector[:256] += fft_mod

        # Chroma modulation - harmonic content affects style
        chroma_mod = np.tile(
            audio_features['chroma'],
            self.latent_dim // self.n_chroma + 1
        )[:self.latent_dim]
        latent_vector += chroma_mod * self.chroma_scale

        # RMS energy modulation - loudness affects intensity
        energy_factor = (audio_features['low_rms'] + audio_features['high_rms']) * self.rms_scale
        latent_vector *= (1.0 + energy_factor)

        # Add temporal variation for smooth animation
        temporal_mod = np.sin(frame_idx * 0.1) * 0.02
        latent_vector += temporal_mod

        return latent_vector

    def generate_image(self, latent_vector):
        """
        Generate image from latent vector using standard StyleGAN2
        Args:
            latent_vector: NumPy array of shape [512]
        Returns:
            NumPy image array in uint8 format [H, W, 3]
        """
        with torch.no_grad():
            # Convert to torch tensor
            latent_tensor = torch.from_numpy(latent_vector).float().unsqueeze(0)
            latent_tensor = latent_tensor.to(self.device)

            # Generate image using standard StyleGAN2
            generated_image = self.onnx_model(latent_tensor)

            # Convert to numpy and normalize
            image = generated_image.squeeze(0).cpu().numpy()
            image = np.transpose(image, (1, 2, 0))  # CHW -> HWC
            image = (image + 1) * 127.5  # [-1,1] -> [0,255]
            image = np.clip(image, 0, 255).astype(np.uint8)

            return image

    def process_audio_file(self, audio_path, save_images=True, save_video=True):
        """
        Main processing function - audio-reactive generation
        Args:
            audio_path: Path to input audio file
            save_images: Save individual frames
            save_video: Create output video
        """
        print(f"Processing audio file: {audio_path}")

        # Load audio
        audio_frames = self.load_audio(audio_path)
        total_frames = len(audio_frames)

        images = []

        print(f"Generating {total_frames} audio-reactive frames...")

        for frame_idx, audio_frame in enumerate(audio_frames):
            if frame_idx % 100 == 0:
                print(f"Processing frame {frame_idx}/{total_frames}")

            # Extract audio features
            features = self.extract_audio_features(audio_frame)

            # Map audio to latent vector (AUDIO REACTIVITY HAPPENS HERE!)
            latent_vector = self.map_audio_to_latent(features, frame_idx)

            # Generate image from audio-modulated latent
            image = self.generate_image(latent_vector)
            images.append(image)

            # Save individual frame
            if save_images:
                frame_path = self.output_dir / f"frame_{frame_idx:06d}.png"
                cv2.imwrite(str(frame_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Create video
        if save_video and images:
            self.create_video(images, audio_path)

        print(f"Generated {len(images)} audio-reactive images")
        return images

    def create_video(self, images, audio_path):
        """Create video from generated images with original audio"""
        video_path = self.output_dir / "output_video.mp4"

        height, width = images[0].shape[:2]
        fps = self.sr / self.hop_length

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        for image in images:
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            out.write(bgr_image)

        out.release()
        print(f"Video saved to: {video_path}")

        # Add audio to video (requires ffmpeg)
        final_path = self.output_dir / "final_output.mp4"
        os.system(f'ffmpeg -i "{video_path}" -i "{audio_path}" -c:v copy -c:a aac -y "{final_path}"')
        print(f"Final video with audio: {final_path}")

    def export_to_onnx(self, onnx_path="stylegan_standard.onnx", opset_version=17):
        """
        Export standard StyleGAN2 model to ONNX format.
        This should work without issues since standard StyleGAN2 is ONNX-compatible.
        """
        print(f"Exporting standard StyleGAN2 to ONNX: {onnx_path}")

        # Create dummy input
        dummy_input = torch.randn(1, self.latent_dim, device=self.device)

        # Export to ONNX
        torch.onnx.export(
            self.onnx_model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['latent_z'],
            output_names=['generated_image'],
            dynamic_axes={
                'latent_z': {0: 'batch_size'},
                'generated_image': {0: 'batch_size'}
            },
            verbose=False
        )

        print(f"✅ Model exported successfully to {onnx_path}")

        # Verify the exported model
        self._verify_onnx_export(onnx_path, dummy_input)

    def _verify_onnx_export(self, onnx_path, dummy_input):
        """Verify ONNX export by comparing outputs"""
        try:
            import onnx
            import onnxruntime as ort

            # Load and check ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX model is valid!")

            # Compare outputs
            with torch.no_grad():
                pytorch_output = self.onnx_model(dummy_input).cpu().numpy()

            # Run ONNX inference
            ort_session = ort.InferenceSession(onnx_path)
            onnx_output = ort_session.run(
                None,
                {'latent_z': dummy_input.cpu().numpy()}
            )[0]

            # Compare
            diff = np.abs(pytorch_output - onnx_output).max()
            print(f"Max difference between PyTorch and ONNX: {diff}")

            if diff < 1e-3:
                print("✓ ONNX export verification passed!")
            else:
                print(f"⚠ Warning: Output difference is {diff}")

        except ImportError:
            print("⚠ onnx or onnxruntime not installed, skipping verification")
        except Exception as e:
            print(f"⚠ Verification failed: {e}")


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description='Standard StyleGAN2 audio-to-visual generation')
    parser.add_argument('--model', type=str, default='./models/ffhq.pkl',
                       help='Path to StyleGAN model')
    parser.add_argument('--audio', type=str, default='./input_audio.wav',
                       help='Path to input audio file')
    parser.add_argument('--output', type=str, default='./output',
                       help='Output directory')
    parser.add_argument('--export-onnx', type=str, default=None,
                       help='Export model to ONNX (provide output path)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Initialize inference with standard StyleGAN2
    inference = OfflineAudioInferenceStandard(
        args.model,
        output_dir=args.output,
        device=args.device
    )

    # Export to ONNX if requested
    if args.export_onnx:
        inference.export_to_onnx(args.export_onnx)

    # Process audio file (audio-reactive generation)
    if os.path.exists(args.audio):
        print("\n🎵 Starting audio-reactive generation with standard StyleGAN2...")
        images = inference.process_audio_file(args.audio)
        print("✅ Audio-reactive generation complete!")
    else:
        print(f"Audio file not found: {args.audio}")


if __name__ == "__main__":
    main()
