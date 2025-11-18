#!/usr/bin/env python3
"""
Offline Audio-to-Visual Inference Script
Processes an audio file and generates images using StyleGAN without GUI
"""

import numpy as np
import torch
import librosa
import cv2
import os
from pathlib import Path

# Autolume imports
from widgets.renderer import Renderer
from audio.feature_extractor import FeatureExtractor
import dnnlib


class OfflineAudioInference:
    def __init__(self, model_path, output_dir="./output"):
        """
        Initialize offline inference
        Args:
            model_path: Path to trained StyleGAN pickle file
            output_dir: Directory to save generated images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize renderer (loads StyleGAN model)
        self.renderer = Renderer()
        self.renderer.set_pkl(model_path)

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
        # These would typically come from the GUI widgets in real-time mode
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
            hop_length=self.hop_length  # No overlap - matches audio duration
        )

        return frames.T  # Shape: (n_frames, frame_length)

    def extract_audio_features(self, audio_frame):
        """Extract features from single audio frame"""
        # FFT analysis
        fft = np.abs(librosa.stft(audio_frame, n_fft=self.n_fft * 2 - 1))
        fft_features = fft.mean(axis=1)  # Average over time

        # Chroma features
        chroma = librosa.feature.chroma_stft(S=fft**2, n_chroma=self.n_chroma)
        chroma_features = chroma.mean(axis=1)
        chroma_features /= (chroma_features.sum() + 1e-8)  # Normalize

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
        """Map audio features to StyleGAN latent vector"""
        # Generate base latent vector from seed
        np.random.seed(self.base_seed)
        base_latent = np.random.randn(self.latent_dim)

        # Modulate latent vector with audio features
        latent_vector = base_latent.copy()

        # FFT modulation - use first 256 components to modulate latent
        fft_mod = audio_features['fft'][:256] * self.fft_scale
        latent_vector[:256] += fft_mod

        # Chroma modulation - cycle through latent dimensions
        chroma_mod = np.tile(audio_features['chroma'],
                           self.latent_dim // self.n_chroma + 1)[:self.latent_dim]
        latent_vector += chroma_mod * self.chroma_scale

        # RMS energy modulation - global scaling
        energy_factor = (audio_features['low_rms'] + audio_features['high_rms']) * self.rms_scale
        latent_vector *= (1.0 + energy_factor)

        # Add temporal variation
        temporal_mod = np.sin(frame_idx * 0.1) * 0.02
        latent_vector += temporal_mod

        return latent_vector

    def generate_image(self, latent_vector):
        """Generate image from latent vector using StyleGAN"""
        with torch.no_grad():
            # Convert to torch tensor (Z latent space: shape [1, 512])
            latent_tensor = torch.from_numpy(latent_vector).float().unsqueeze(0)

            if torch.cuda.is_available():
                latent_tensor = latent_tensor.cuda()

            # Following "vec mode with project=True" approach from renderer.py lines 565-566
            # Pass Z through mapping network to get W+ latent
            mapping_net = self.renderer.G.mapping
            all_cs = torch.zeros([1, self.renderer.G.c_dim], dtype=torch.float32)
            if torch.cuda.is_available():
                all_cs = all_cs.cuda()

            # Map Z → W+ (shape: [1, 512] → [1, num_ws, 512])
            w_latent = mapping_net(
                z=latent_tensor,
                c=all_cs,
                truncation_psi=0.7,
                truncation_cutoff=8
            )

            # Generate image through StyleGAN synthesis network
            generated_image, _ = self.renderer.run_synthesis_net(
                w_latent,
                noise_mode='const',
                force_fp32=True
            )

            # Convert to numpy and normalize
            image = generated_image.squeeze(0).cpu().numpy()
            image = np.transpose(image, (1, 2, 0))  # CHW -> HWC
            image = (image + 1) * 127.5  # [-1,1] -> [0,255]
            image = np.clip(image, 0, 255).astype(np.uint8)

            return image

    def process_audio_file(self, audio_path, save_images=True, save_video=True):
        """
        Main processing function
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

        print(f"Generating {total_frames} frames...")

        for frame_idx, audio_frame in enumerate(audio_frames):
            if frame_idx % 100 == 0:
                print(f"Processing frame {frame_idx}/{total_frames}")

            # Extract audio features
            features = self.extract_audio_features(audio_frame)

            # Map to latent vector
            latent_vector = self.map_audio_to_latent(features, frame_idx)

            # Generate image
            image = self.generate_image(latent_vector)
            images.append(image)

            # Save individual frame
            if save_images:
                frame_path = self.output_dir / f"frame_{frame_idx:06d}.png"
                cv2.imwrite(str(frame_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Create video
        if save_video and images:
            self.create_video(images, audio_path)

        print(f"Generated {len(images)} images")
        return images

    def create_video(self, images, audio_path):
        """Create video from generated images with original audio"""
        video_path = self.output_dir / "output_video.mp4"

        height, width = images[0].shape[:2]
        # Calculate FPS to match audio duration: sample_rate / hop_length
        fps = self.sr / self.hop_length  # 44100 / 1024 ≈ 43 fps

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        for image in images:
            # Convert RGB to BGR for OpenCV
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            out.write(bgr_image)

        out.release()
        print(f"Video saved to: {video_path}")

        # Add audio to video (requires ffmpeg)
        final_path = self.output_dir / "final_output.mp4"
        os.system(f'ffmpeg -i "{video_path}" -i "{audio_path}" -c:v copy -c:a aac -y "{final_path}"')
        print(f"Final video with audio: {final_path}")


def main():
    """Example usage"""
    # Initialize inference
    model_path = "./models/ffhq.pkl"  # Update this path
    inference = OfflineAudioInference(model_path)

    # Process audio file
    audio_path = "./input_audio.wav"  # Update this path
    images = inference.process_audio_file(audio_path)

    print("Audio-to-visual generation complete!")
    print("hello world")

if __name__ == "__main__":
    main()