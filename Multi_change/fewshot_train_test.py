#!/usr/bin/env python3
"""
Simple few-shot fine-tuning script for HPC batch jobs.

Trains a model on N% of data, tests it, and outputs metrics to CSV.

Usage:
    python fewshot_train_test.py \
        --checkpoint ./models_ckpt/benchmarking_ckpts/model.pth \
        --train_script train.py \
        --data_pct 25 \
        --output_dir ./models_ckpt/few-shot-experiments \
        --encoder_dim 512

Output:
    - {output_dir}/checkpoint.pth
    - {output_dir}/metrics.csv (mIoU, IoU per class)
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

from utils_tool.utils import *


def run_training(args):
    """Run training and return best checkpoint path."""
    print(f"\n{'='*60}")
    print(f"TRAINING: {args.data_pct}% data")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Script: {args.train_script}")
    print(f"{'='*60}\n")

    # Build training command
    cmd = [
        "python",
        args.train_script,
        "--data_name",
        args.dataname,
        "--data_folder",
        args.data_folder,
        "--checkpoint",
        args.checkpoint,
        "--train_batchsize",
        str(args.batch_size),
        "--num_epochs",
        str(args.num_epochs),
        "--patience",
        str(args.patience),
        "--workers",
        str(args.workers),
        "--savepath",
        str(args.output_dir),
        "--max_percent_samples",
        str(args.data_pct),
    ]

    # Add script-specific args
    if args.train_script == "train_benchmark.py":
        cmd.extend(["--benchmark", args.benchmark, "--train_goal", "0"])
    else:
        cmd.extend(
            [
                "--train_goal",
                "0",
            ]
        )

    # Run
    log_file = args.output_dir / "train.log"
    print(f"Command: {' '.join(cmd)}")
    print(f"Log: {log_file}\n")

    with open(log_file, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"❌ Training failed! Check {log_file}")
        sys.exit(1)

    # Find best checkpoint
    checkpoints = list(args.output_dir.glob("**/*.pth"))
    if not checkpoints:
        print(f"❌ No checkpoint found in {args.output_dir}")
        sys.exit(1)

    best_ckpt = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"✅ Training complete: {best_ckpt.name}\n")
    return best_ckpt


def run_testing(args, checkpoint):
    """Run testing and parse metrics."""
    print(f"\n{'='*60}")
    print(f"TESTING")
    print(f"{'='*60}\n")

    test_dir = args.output_dir / "test_results"
    test_dir.mkdir(exist_ok=True)

    # Determine test script
    test_script = (
        "test_benchmark.py" if args.train_script == "train_benchmark.py" else "test.py"
    )

    # Build test command
    cmd = [
        "python",
        test_script,
        "--data_name",
        args.dataname,
        "--data_folder",
        args.data_folder,
        "--checkpoint",
        str(checkpoint),
        "--test_goal",
        "0",
        "--result_path",
        str(test_dir),
        "--workers",
        str(args.workers),
    ]

    if args.train_script == "train_benchmark.py":
        cmd.extend(["--benchmark", args.benchmark])
    else:
        cmd.extend(["--network", args.network, "--encoder_dim", str(args.encoder_dim)])

    # Run
    log_file = test_dir / "test.log"
    print(f"Command: {' '.join(cmd)}")
    print(f"Log: {log_file}\n")

    with open(log_file, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"❌ Testing failed! Check {log_file}")
        sys.exit(1)

    # Parse metrics
    metrics = parse_metrics(log_file)
    print(f"✅ Testing complete\n")
    return metrics


def parse_metrics(log_file):
    """Parse mIoU and IoU string from test log."""
    with open(log_file, "r") as f:
        content = f.read()

    metrics = {}

    # Parse mIoU
    match = re.search(r"mIoU_seg:\s+([\d.]+)", content)
    if match:
        metrics["mIoU"] = float(match.group(1))
    else:
        print("⚠️ Could not parse mIoU")
        metrics["mIoU"] = None

    # Parse full IoU string (keep it as-is)
    match = re.search(r"IoU:\s+([0-9.\s]+)", content)
    if match:
        metrics["IoU"] = match.group(1).strip()
    else:
        print("⚠️ Could not parse IoU")
        metrics["IoU"] = None

    return metrics


def save_metrics(args, metrics, checkpoint):
    """Save metrics to CSV."""
    csv_file = args.output_dir / "metrics.csv"

    # Prepare row
    row = {
        "checkpoint": Path(args.checkpoint).name,
        "max_percent_samples": args.data_pct,
        "network": args.network if args.train_script == "train.py" else args.benchmark,
        "mIoU": metrics.get("mIoU"),
        "IoU": metrics.get("IoU"),
    }

    row["trained_checkpoint"] = str(checkpoint)

    # Write CSV
    import csv

    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    for key, value in row.items():
        if key not in ["checkpoint", "trained_checkpoint"]:
            print(f"{key:20s}: {value}")
    print(f"{'='*60}")
    print(f"✅ Metrics saved to: {csv_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Few-shot fine-tuning: train + test in one script"
    )

    # Required
    parser.add_argument("--checkpoint", required=True, help="Path to source checkpoint")
    parser.add_argument(
        "--data_pct",
        type=int,
        required=True,
        help="Percentage of training data (5, 10, 25, 50, 100)",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Output directory for this experiment"
    )

    # Dataset
    parser.add_argument("--dataname", default="JL1-CD-Trees", help="Dataset name")
    parser.add_argument(
        "--data_folder", default="./JL1-CD-Trees/images", help="Data folder"
    )

    # Model
    parser.add_argument(
        "--train_script", default="train.py", choices=["train.py", "train_benchmark.py"]
    )
    parser.add_argument(
        "--network", default="segformer-mit_b1", help="Network (for train.py)"
    )
    parser.add_argument(
        "--encoder_dim", type=int, default=512, help="Encoder dim (for train.py)"
    )
    parser.add_argument(
        "--benchmark", default="bifa", help="Benchmark model (for train_benchmark.py)"
    )
    parser.add_argument(
        "--fine_tune_encoder",
        type=str2bool,
        default=True,
        help="whether fine-tune encoder or not",
    )

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=0)

    args = parser.parse_args()

    # Setup
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Train
    checkpoint = run_training(args)

    # Test
    metrics = run_testing(args, checkpoint)

    # Save
    save_metrics(args, metrics, checkpoint)

    print("✅ Done!\n")


if __name__ == "__main__":
    main()
