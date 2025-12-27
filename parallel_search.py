#!/usr/bin/env python3
"""
Parallel Hyperparameter Search Launcher

Runs multiple single-GPU hyperparameter search processes in parallel,
one trial per GPU. All processes share the same Optuna study database
for coordinated parallel optimization.

Usage:
    python parallel_search.py \\
        --data_root /path/to/data \\
        --checkpoint_dir /path/to/checkpoints \\
        --n_trials 24 \\
        --use_all_writers

With 4 GPUs and 24 trials, each GPU will run ~6 trials (24/4).
All trials are coordinated through shared SQLite database.
"""

import os
import sys
import time
import signal
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

import torch


class ParallelSearchLauncher:
    """Manages parallel hyperparameter search across multiple GPUs."""

    def __init__(self, args):
        self.args = args
        self.processes = []
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        if self.num_gpus == 0:
            raise RuntimeError("No GPUs available. Use single_gpu_hyperparameter_search.py directly.")

        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C and termination signals."""
        print("\n\n" + "="*70)
        print("🛑 INTERRUPT RECEIVED - Gracefully stopping all workers...")
        print("="*70)
        self._cleanup()
        sys.exit(0)

    def _cleanup(self):
        """Terminate all child processes."""
        for i, proc in enumerate(self.processes):
            if proc.poll() is None:  # Process still running
                print(f"  Stopping worker on GPU {i}...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"  Force killing worker on GPU {i}...")
                    proc.kill()

        self.processes = []
        print("✅ All workers stopped")

    def _create_worker_command(self, gpu_id):
        """Create command for a single worker process."""
        # Path to single GPU script
        script_path = Path(__file__).parent / "single_gpu_hyperparameter_search.py"

        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        # Build command
        cmd = [
            sys.executable,  # Python interpreter
            str(script_path),
            "--data_root", self.args.data_root,
            "--checkpoint_dir", self.args.checkpoint_dir,
            "--n_trials", str(self.args.n_trials),
        ]

        if self.args.use_all_writers:
            cmd.append("--use_all_writers")
        else:
            cmd.extend(["--num_writers_subset", str(self.args.num_writers_subset)])

        return cmd

    def _create_log_file(self, gpu_id):
        """Create log file for worker output."""
        log_dir = Path(self.args.checkpoint_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"worker_gpu{gpu_id}_{timestamp}.log"

        return log_file

    def launch(self):
        """Launch parallel workers."""
        print("\n" + "="*70)
        print("PARALLEL HYPERPARAMETER SEARCH LAUNCHER")
        print("="*70)
        print(f"Available GPUs: {self.num_gpus}")
        print(f"Total trials: {self.args.n_trials}")
        print(f"Trials per GPU: ~{self.args.n_trials // self.num_gpus}")
        print(f"Checkpoint dir: {self.args.checkpoint_dir}")
        print(f"Data root: {self.args.data_root}")
        print("="*70)

        # Create checkpoint directory
        Path(self.args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        print("\n🚀 Launching workers...\n")

        # Launch one worker per GPU
        for gpu_id in range(self.num_gpus):
            cmd = self._create_worker_command(gpu_id)
            log_file = self._create_log_file(gpu_id)

            # Set environment to restrict to single GPU
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            # Open log file
            log_handle = open(log_file, 'w')

            # Launch process
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1  # Line buffered
            )

            self.processes.append(proc)

            print(f"  ✅ GPU {gpu_id}: Worker launched (PID {proc.pid})")
            print(f"     Log: {log_file}")

            # Small delay to stagger database access
            time.sleep(2)

        print("\n" + "="*70)
        print("ALL WORKERS LAUNCHED")
        print("="*70)
        print("\n📊 Monitor progress:")
        print(f"  • Check logs: {Path(self.args.checkpoint_dir) / 'logs'}")
        print(f"  • Check trials: {Path(self.args.checkpoint_dir) / 'trial_*'}")
        print(f"  • Database: {Path(self.args.checkpoint_dir) / 'optuna_study.db'}")
        print("\n💡 Tips:")
        print("  • Each worker pulls trials from shared Optuna database")
        print("  • Workers run independently and may finish at different times")
        print("  • Press Ctrl+C to stop all workers gracefully")
        print("  • If interrupted, resume with the same command - Optuna will continue")
        print("\n⏳ Waiting for workers to complete...\n")

    def wait_for_completion(self):
        """Wait for all workers to complete."""
        completed = []

        try:
            while len(completed) < self.num_gpus:
                for i, proc in enumerate(self.processes):
                    if i in completed:
                        continue

                    # Check if process finished
                    retcode = proc.poll()
                    if retcode is not None:
                        completed.append(i)
                        if retcode == 0:
                            print(f"  ✅ GPU {i} worker completed successfully")
                        else:
                            print(f"  ⚠️  GPU {i} worker exited with code {retcode}")

                # Sleep briefly to avoid busy waiting
                time.sleep(5)

        except KeyboardInterrupt:
            self._signal_handler(None, None)

        print("\n" + "="*70)
        print("ALL WORKERS COMPLETED")
        print("="*70)

        # Check for failures
        failures = [i for i, proc in enumerate(self.processes) if proc.returncode != 0]
        if failures:
            print(f"\n⚠️  Some workers failed: GPUs {failures}")
            print("   Check logs for details")
        else:
            print("\n✅ All workers completed successfully!")

        print(f"\n📁 Results saved to: {self.args.checkpoint_dir}")
        print(f"   • Summary: {Path(self.args.checkpoint_dir) / 'summary'}")
        print(f"   • Best model: {Path(self.args.checkpoint_dir) / 'best_overall'}")
        print("="*70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Parallel Hyperparameter Search - Run one trial per GPU in parallel',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 24 trials across 4 GPUs (6 trials per GPU)
  python parallel_search.py \\
    --data_root /path/to/data \\
    --checkpoint_dir ./checkpoints \\
    --n_trials 24 \\
    --use_all_writers

  # Resume interrupted search (same command)
  python parallel_search.py \\
    --data_root /path/to/data \\
    --checkpoint_dir ./checkpoints \\
    --n_trials 24 \\
    --use_all_writers

Notes:
  • All GPUs share the same Optuna study database
  • Each GPU runs one trial at a time
  • Trials are distributed automatically by Optuna
  • Search is fully resumable if interrupted
  • Total time ≈ (n_trials / num_gpus) × time_per_trial
        """
    )

    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to Mirath_extracted_lines directory')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory to save checkpoints and results (shared by all workers)')
    parser.add_argument('--n_trials', type=int, default=12,
                       help='Total number of trials to run (distributed across GPUs)')
    parser.add_argument('--use_all_writers', action='store_true',
                       help='Use all writers (default: True)')
    parser.add_argument('--num_writers_subset', type=int, default=7,
                       help='Number of writers if not using all (default: 7)')

    args = parser.parse_args()

    # Validate paths
    if not Path(args.data_root).exists():
        print(f"❌ Error: Data root not found: {args.data_root}")
        sys.exit(1)

    # Create launcher
    launcher = ParallelSearchLauncher(args)

    # Launch workers
    launcher.launch()

    # Wait for completion
    launcher.wait_for_completion()


if __name__ == '__main__':
    main()
