#!/usr/bin/env python3
"""
HuggingFace Jobs Runner for TorchScript StyleGAN2 Workflow

This script runs the complete TorchScript workflow on HuggingFace Jobs:
1. Export fresh Custom StyleGAN2 to TorchScript (.pt file)
2. Load the TorchScript model
3. Generate audio-reactive video
4. Upload all artifacts to HuggingFace repo

Usage:
    # Export + inference + upload
    python run_torchscript_job.py \
        --image ayh2bxa/autolume \
        --audio-path /app/autolume/input_audio.wav \
        --output-repo username/autolume-outputs \
        --monitor

    # Export only
    python run_torchscript_job.py \
        --image ayh2bxa/autolume \
        --export-only \
        --output-repo username/autolume-models
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import run_job, list_jobs, fetch_job_logs, HfApi
    from huggingface_hub.errors import RepositoryNotFoundError, HTTPError
except ImportError:
    print("Error: huggingface_hub not installed. Install with: pip install huggingface_hub")
    exit(1)


class TorchScriptJobRunner:
    """Manages HuggingFace jobs for TorchScript export and inference"""

    def __init__(self, token: Optional[str] = None):
        """
        Initialize the job runner

        Args:
            token: Hugging Face token (optional, uses environment or cached token)
        """
        self.api = HfApi(token=token)
        self.token = token

    def run_torchscript_job(
        self,
        docker_image: str,
        audio_path: Optional[str] = None,
        export_only: bool = False,
        output_repo: Optional[str] = None,
        torchscript_path: str = "/tmp/stylegan2.pt",
        hardware_flavor: str = "l4x1",
        resolution: int = 512,
        max_frames: Optional[int] = None,
        env_vars: Optional[dict] = None
    ):
        """
        Run TorchScript export and/or inference job on HuggingFace infrastructure

        Args:
            docker_image: Docker image URL (e.g., "ayh2bxa/autolume:latest")
            audio_path: Path to input audio file in container (required if not export_only)
            export_only: Only export TorchScript model (no inference)
            output_repo: HF repo to upload results (optional)
            torchscript_path: Path to save TorchScript model
            hardware_flavor: HF hardware flavor
            resolution: Image resolution (512 or 1024)
            max_frames: Maximum frames to generate (None = all)
            env_vars: Additional environment variables
        """

        # Ensure image has proper registry prefix
        if not docker_image.startswith(('docker.io/', 'registry.hf.space/', 'ghcr.io/')):
            docker_image = f"docker.io/{docker_image}"

        print(f"🚀 Starting TorchScript job...")
        print(f"   Image: {docker_image}")
        print(f"   Mode: {'Export only' if export_only else 'Export + Inference'}")
        if audio_path:
            print(f"   Audio: {audio_path}")
        print(f"   TorchScript output: {torchscript_path}")
        print(f"   Hardware: {hardware_flavor}")
        print(f"   Resolution: {resolution}x{resolution}")

        # Build Python script
        if export_only:
            python_script = self._build_export_script(
                torchscript_path, resolution, output_repo
            )
        else:
            if not audio_path:
                raise ValueError("audio_path is required when export_only=False")
            python_script = self._build_full_workflow_script(
                torchscript_path, audio_path, resolution, max_frames, output_repo
            )

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
                token=self.token
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

    def _build_export_script(self, torchscript_path, resolution, output_repo):
        """Build Python script for TorchScript export only"""
        return f"""
from torchscript_inference import TorchScriptStyleGAN2Wrapper
from huggingface_hub import HfApi
import os

def main():
    print('📦 TorchScript Export')
    print('=' * 70)
    print()

    torchscript_path = '{torchscript_path}'
    output_repo = '{output_repo}'
    resolution = {resolution}

    print(f'TorchScript output: {{torchscript_path}}')
    print(f'Resolution: {{resolution}}x{{resolution}}')
    print()

    # Create wrapper and export
    print('🔧 Initializing wrapper...')
    wrapper = TorchScriptStyleGAN2Wrapper(
        img_resolution=resolution,
        device='cuda'
    )

    print('\\n📦 Exporting to TorchScript...')
    saved_path = wrapper.export_to_torchscript(torchscript_path)

    print(f'\\n✅ TorchScript export complete!')
    print(f'   File: {{saved_path}}')

    # Upload to HuggingFace if repo specified
    if output_repo and output_repo != 'None':
        print(f'\\n📤 Uploading to {{output_repo}}...')
        hf_token = os.environ.get('HF_TOKEN')
        api = HfApi(token=hf_token)

        try:
            api.upload_file(
                path_or_fileobj=saved_path,
                path_in_repo=os.path.basename(saved_path),
                repo_id=output_repo,
                repo_type='model'
            )
            print(f'   ✅ Uploaded to https://huggingface.co/{{output_repo}}')
        except Exception as e:
            print(f'   ⚠️  Upload failed: {{e}}')

if __name__ == '__main__':
    main()
"""

    def _build_full_workflow_script(
        self,
        torchscript_path,
        audio_path,
        resolution,
        max_frames,
        output_repo
    ):
        """Build Python script for full TorchScript workflow (export + inference)"""
        max_frames_str = str(max_frames) if max_frames is not None else 'None'

        return f"""
from torchscript_inference import TorchScriptStyleGAN2Wrapper
from huggingface_hub import HfApi
import os

def main():
    print('🚀 TorchScript Full Workflow')
    print('=' * 70)
    print()

    torchscript_path = '{torchscript_path}'
    audio_path = '{audio_path}'
    output_repo = '{output_repo}'
    resolution = {resolution}
    max_frames = {max_frames_str}

    print(f'TorchScript path: {{torchscript_path}}')
    print(f'Audio: {{audio_path}}')
    print(f'Resolution: {{resolution}}x{{resolution}}')
    if max_frames:
        print(f'Max frames: {{max_frames}}')
    print()

    # Step 1: Export to TorchScript
    print('\\n📦 Step 1: Export to TorchScript')
    print('-' * 70)

    wrapper = TorchScriptStyleGAN2Wrapper(
        img_resolution=resolution,
        device='cuda'
    )

    saved_path = wrapper.export_to_torchscript(torchscript_path)
    print(f'   ✓ TorchScript model saved: {{saved_path}}')

    # Step 2: Load TorchScript model
    print('\\n📂 Step 2: Load TorchScript Model')
    print('-' * 70)

    loaded_model = wrapper.load_torchscript(saved_path)
    print(f'   ✓ Model loaded successfully')

    # Step 3: Generate audio-reactive video
    print('\\n🎵 Step 3: Audio-Reactive Video Generation')
    print('-' * 70)

    video_path, frames_dir = wrapper.process_audio_with_torchscript(
        loaded_model,
        audio_path,
        output_dir='/tmp/output',
        max_frames=max_frames,
        fps=30
    )

    print(f'\\n✅ Video generation complete!')
    print(f'   Video: {{video_path}}')
    print(f'   Frames: {{frames_dir}}')

    # Step 4: Upload to HuggingFace
    if output_repo and output_repo != 'None':
        print('\\n📤 Step 4: Upload to HuggingFace')
        print('-' * 70)

        hf_token = os.environ.get('HF_TOKEN')
        api = HfApi(token=hf_token)

        files_to_upload = [
            (saved_path, 'TorchScript model'),
            (video_path, 'Generated video')
        ]

        for filepath, description in files_to_upload:
            if os.path.exists(filepath):
                try:
                    filename = os.path.basename(filepath)
                    api.upload_file(
                        path_or_fileobj=filepath,
                        path_in_repo=filename,
                        repo_id=output_repo,
                        repo_type='model'
                    )
                    print(f'   ✓ Uploaded {{description}}: {{filename}}')
                except Exception as e:
                    print(f'   ⚠️  Failed to upload {{description}}: {{e}}')

        print(f'\\n🎉 All files uploaded to https://huggingface.co/{{output_repo}}/tree/main')

    print('\\n' + '=' * 70)
    print('✅ Complete!')
    print(f'   TorchScript model: {{saved_path}}')
    print(f'   Video: {{video_path}}')
    print('=' * 70)

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

        print(f"\\n📊 Job submitted! Monitoring status...")
        print(f"   Job ID: {job_id}")
        print(f"   View in browser: https://huggingface.co/jobs/{job_id}")
        print(f"\\n💡 Tip: Press Ctrl+C to stop monitoring (job will continue running)")
        print("=" * 70)

        last_log_length = 0

        try:
            while True:
                # Fetch logs
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
                            print("\\n" + "=" * 70)
                            print("🏁 Job appears to have finished")
                            print(f"   View full results: https://huggingface.co/jobs/{job_id}")
                            break
                    else:
                        print(".", end='', flush=True)

                except Exception as e:
                    print(f"\\n⚠️  Could not fetch logs: {e}")
                    print(f"   Check status at: https://huggingface.co/jobs/{job_id}")
                    break

                # Wait before next check
                time.sleep(10)

        except KeyboardInterrupt:
            print("\\n" + "=" * 70)
            print("⏸️  Monitoring stopped")
            print(f"   Job continues running at: https://huggingface.co/jobs/{job_id}")
        except Exception as e:
            print(f"\\n❌ Error during monitoring: {e}")
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
        description="Run TorchScript workflow on Hugging Face Jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export + inference + upload
  python run_torchscript_job.py \\
      --image ayh2bxa/autolume \\
      --audio-path /app/autolume/input_audio.wav \\
      --output-repo username/autolume-outputs \\
      --monitor

  # Export only
  python run_torchscript_job.py \\
      --image ayh2bxa/autolume \\
      --export-only \\
      --output-repo username/autolume-models

  # Custom resolution and frame limit
  python run_torchscript_job.py \\
      --image ayh2bxa/autolume \\
      --audio-path /app/input.wav \\
      --resolution 1024 \\
      --max-frames 100
        """
    )

    # Required arguments
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Docker image URL (e.g., ayh2bxa/autolume:latest)"
    )

    # Mode selection
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export TorchScript model (no inference)"
    )

    # Audio path
    parser.add_argument(
        "--audio-path", "-a",
        help="Path to input audio file in container (required if not --export-only)"
    )

    # Output options
    parser.add_argument(
        "--output-repo", "-o",
        help="HuggingFace repo to upload outputs (e.g., username/repo-name)"
    )

    parser.add_argument(
        "--torchscript-path",
        default="/tmp/stylegan2.pt",
        help="Path to save TorchScript model (default: /tmp/stylegan2.pt)"
    )

    # Model settings
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        choices=[512, 1024],
        help="Image resolution (default: 512)"
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum frames to generate (default: all)"
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
    runner = TorchScriptJobRunner(token=args.token)

    # List jobs if requested
    if args.list_jobs:
        runner.list_recent_jobs()
        return

    # Validate arguments
    if not args.export_only and not args.audio_path:
        parser.error("--audio-path is required when not using --export-only")

    # Submit job
    job = runner.run_torchscript_job(
        docker_image=args.image,
        audio_path=args.audio_path,
        export_only=args.export_only,
        output_repo=args.output_repo,
        torchscript_path=args.torchscript_path,
        hardware_flavor=args.hardware,
        resolution=args.resolution,
        max_frames=args.max_frames
    )

    if job and args.monitor:
        job_id = job.job_id if hasattr(job, 'job_id') else str(job)
        runner.monitor_job(job_id, follow_logs=True)


if __name__ == "__main__":
    main()
