#!/usr/bin/env python3
"""
Hugging Face Jobs for Audio-Reactive Custom StyleGAN2

This script uses Hugging Face's run_job function to:
1. Create a FRESH (untrained) Custom StyleGAN2 model
2. Generate audio-reactive video

Uses: offline_audio_inference.py
- Custom StyleGAN2 architecture (custom_stylegan2.py)
- NO pretrained weights needed
- Full audio reactivity maintained
- Square images only (512x512 or 1024x1024)

Prerequisites:
- Hugging Face Pro account or Team/Enterprise organization
- Docker image pushed to Docker Hub or HF registry (AMD64 architecture)
- HF token configured (hf login or HF_TOKEN env var)

Usage:
    # Run audio-reactive inference
    python run_audio_inference_job.py --image ayh2bxa/autolume --audio-path /app/autolume/input_audio.wav

    # Upload results to HF repo
    python run_audio_inference_job.py --image ayh2bxa/autolume --audio-path /app/autolume/input_audio.wav --output-repo username/repo-name
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import run_job, list_jobs, inspect_job, fetch_job_logs, HfApi
    from huggingface_hub.errors import RepositoryNotFoundError, HTTPError
except ImportError:
    print("Error: huggingface_hub not installed. Install with: pip install huggingface_hub")
    exit(1)


class AudioInferenceJobRunner:
    """Manages Hugging Face jobs for audio-reactive inference"""

    def __init__(self, token: Optional[str] = None):
        """
        Initialize the job runner

        Args:
            token: Hugging Face token (optional, uses environment or cached token)
        """
        self.api = HfApi(token=token)
        self.token = token

    def run_inference_job(
        self,
        docker_image: str,
        audio_path: str,
        output_repo: Optional[str] = None,
        hardware_flavor: str = "l4x1",
        env_vars: Optional[dict] = None
    ):
        """
        Run audio-reactive inference job on Hugging Face infrastructure

        Args:
            docker_image: Docker image URL (e.g., "ayh2/autolume:latest")
            audio_path: Path to input audio file in container
            output_repo: HF repo to upload results (optional)
            hardware_flavor: HF hardware flavor
            env_vars: Additional environment variables
        """

        # Ensure image has proper registry prefix
        if not docker_image.startswith(('docker.io/', 'registry.hf.space/', 'ghcr.io/')):
            docker_image = f"docker.io/{docker_image}"

        print(f"🚀 Starting audio-reactive inference job...")
        print(f"   Image: {docker_image}")
        print(f"   Audio: {audio_path}")
        print(f"   Hardware: {hardware_flavor}")

        # Build Python script for inference
        if not audio_path:
            raise ValueError("audio_path is required")

        python_script = self._build_inference_script(audio_path, output_repo)

        # Prepare command to run in container
        command = [
            "/bin/bash", "-c", f"""
cd /app/autolume

# Check if venv exists, if so activate it
if [ -f venv/bin/activate ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
    echo "   Python: $(which python)"
    echo "   Python version: $(python --version)"
    pip list | grep -E "(torch|librosa)" || echo "⚠️  Some packages not found"
else
    echo "⚠️  venv does not exist at /app/autolume/venv"
fi

# Run the Python script
python -c "
import sys
sys.path.insert(0, '/app/autolume')

{python_script}
"
"""
        ]

        # Setup environment variables
        job_env_vars = {
            "PYTHONPATH": "/app/autolume",
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_TOKEN": self.token or os.environ.get("HF_TOKEN", "")
        }
        if env_vars:
            job_env_vars.update(env_vars)

        try:
            # Submit job
            job = run_job(
                image=docker_image,
                command=command,
                flavor=hardware_flavor,
                env=job_env_vars,
                token=self.token  # Pass token explicitly
            )

            print(f"✅ Job submitted successfully!")
            print(f"   Job ID: {job.job_id if hasattr(job, 'job_id') else job}")

            return job

        except HTTPError as e:
            print(f"❌ HTTP Error: {e}")
            if "401" in str(e):
                print("   Make sure you're logged in: hf login")
            elif "403" in str(e):
                print("   You need a Hugging Face Pro account for Jobs")
            return None
        except Exception as e:
            print(f"❌ Error submitting job: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _build_inference_script(self, audio_path, output_repo):
        """Build Python script for inference only"""
        return f"""
from offline_audio_inference import OfflineAudioInference
from huggingface_hub import HfApi
import os

def main():
    print('🎵 Audio-Reactive Generation (Fresh Custom StyleGAN2)')
    print('=' * 70)

    audio_path = '{audio_path}'
    output_repo = '{output_repo}'

    print(f'Creating fresh Custom StyleGAN2 model (no pretrained weights)')
    print(f'Audio: {{audio_path}}')

    # Create fresh model
    print('\\n🆕 Creating fresh Custom StyleGAN2 model...')
    inference = OfflineAudioInference(
        img_resolution=512,
        output_dir='/tmp/output',
        device='cuda'
    )

    # Process audio
    print('\\n🎨 Processing audio...')
    images = inference.process_audio_file(
        audio_path,
        save_images=True,
        save_video=True
    )

    print(f'\\n✅ Generated {{len(images)}} frames')

    # List output files
    print('\\n📁 Output files:')
    output_files = []
    for root, dirs, files in os.walk('/tmp/output'):
        for file in files:
            filepath = os.path.join(root, file)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f'   {{file}} ({{size_mb:.2f}} MB)')
            output_files.append(filepath)

    # Upload outputs to HuggingFace (only video files)
    if output_repo and output_repo != 'None':
        print(f'\\n📤 Uploading to {{output_repo}}...')
        hf_token = os.environ.get('HF_TOKEN')
        api = HfApi(token=hf_token)

        video_files = [f for f in output_files if f.endswith(('.mp4', '.avi', '.mov'))]

        if not video_files:
            print('⚠️  No video files found to upload')
        else:
            for filepath in video_files:
                filename = os.path.basename(filepath)
                api.upload_file(
                    path_or_fileobj=filepath,
                    path_in_repo=filename,
                    repo_id=output_repo,
                    repo_type='model'
                )
                print(f'   ✅ Uploaded {{filename}}')
            print(f'\\n🎉 All videos uploaded to https://huggingface.co/{{output_repo}}/tree/main')

if __name__ == '__main__':
    main()
"""

    def monitor_job(self, job_id: str, follow_logs: bool = True):
        """
        Monitor job progress and optionally stream logs

        Args:
            job_id: Hugging Face job ID
            follow_logs: Whether to stream logs in real-time
        """

        print(f"\n📊 Job submitted! Monitoring status...")
        print(f"   Job ID: {job_id}")
        print(f"   View in browser: https://huggingface.co/jobs/{job_id}")
        print(f"\n💡 Tip: Press Ctrl+C to stop monitoring (job will continue running)")
        print("=" * 70)

        last_log_length = 0

        try:
            while True:
                # Fetch logs (this also tells us if job is still running)
                try:
                    logs = fetch_job_logs(job_id)

                    if logs:
                        # Only print new logs
                        if len(logs) > last_log_length:
                            new_logs = logs[last_log_length:]
                            print(new_logs, end='', flush=True)
                            last_log_length = len(logs)

                        # Check if logs indicate completion
                        if "Job completed" in logs or "Job failed" in logs or "Error" in logs[-500:]:
                            print("\n" + "=" * 70)
                            print("🏁 Job appears to have finished")
                            print(f"   View full results: https://huggingface.co/jobs/{job_id}")
                            break
                    else:
                        print(".", end='', flush=True)  # Progress indicator

                except Exception as e:
                    # If we can't fetch logs, job might be done or there's an API issue
                    print(f"\n⚠️  Could not fetch logs: {e}")
                    print(f"   Check status at: https://huggingface.co/jobs/{job_id}")
                    break

                # Wait before next check
                time.sleep(10)

        except KeyboardInterrupt:
            print("\n" + "=" * 70)
            print("⏸️  Monitoring stopped")
            print(f"   Job continues running at: https://huggingface.co/jobs/{job_id}")
        except Exception as e:
            print(f"\n❌ Error during monitoring: {e}")
            print(f"   Check job manually at: https://huggingface.co/jobs/{job_id}")

    def list_recent_jobs(self, limit: int = 10):
        """List recent jobs"""

        print(f"📋 Recent jobs (limit: {limit}):")
        try:
            jobs = list_jobs()
            for i, job in enumerate(jobs[:limit]):
                print(f"   {i+1}. {job.job_id} - {job.status} - {job.created_at}")
        except Exception as e:
            print(f"❌ Error listing jobs: {e}")


def main():
    """Main CLI interface"""

    parser = argparse.ArgumentParser(
        description="Run audio-reactive inference on Hugging Face Jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run audio-reactive inference
  python run_audio_inference_job.py --image ayh2bxa/autolume --audio-path /app/autolume/input_audio.wav

  # Upload results to HF repo
  python run_audio_inference_job.py --image ayh2bxa/autolume --audio-path /app/input.wav --output-repo username/repo-name

  # Monitor job progress
  python run_audio_inference_job.py --image ayh2bxa/autolume --audio-path /app/input.wav --monitor
        """
    )

    # Required arguments
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Docker image URL (e.g., ayh2bxa/autolume:latest)"
    )

    parser.add_argument(
        "--audio-path", "-a",
        required=True,
        help="Path to input audio file in container"
    )

    # Output options
    parser.add_argument(
        "--output-repo", "-o",
        help="HuggingFace repo to upload outputs (e.g., username/repo-name)"
    )

    # Hardware settings
    parser.add_argument(
        "--hardware", "-hw",
        default="l4x1",
        help="Hardware flavor (default: l4x1)"
    )

    # Job management
    parser.add_argument(
        "--monitor", "-mon",
        action="store_true",
        help="Monitor job progress and stream logs"
    )

    parser.add_argument(
        "--list-jobs", "-l",
        action="store_true",
        help="List recent jobs and exit"
    )

    parser.add_argument(
        "--token", "-t",
        help="Hugging Face token (uses environment/cached if not provided)"
    )

    args = parser.parse_args()

    # Initialize runner
    runner = AudioInferenceJobRunner(token=args.token)

    # List jobs if requested
    if args.list_jobs:
        runner.list_recent_jobs()
        return

    # Submit job
    job = runner.run_inference_job(
        docker_image=args.image,
        audio_path=args.audio_path,
        output_repo=args.output_repo,
        hardware_flavor=args.hardware
    )

    if job and args.monitor:
        job_id = job.job_id if hasattr(job, 'job_id') else str(job)
        runner.monitor_job(job_id, follow_logs=True)


if __name__ == "__main__":
    main()
