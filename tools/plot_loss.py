#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


LOSS_RE = re.compile(
    r"\((\d+)/\d+\).*?\bloss=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)


def parse_loss(log_path: Path):
    steps = []
    losses = []

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = LOSS_RE.search(line)
            if not match:
                continue
            steps.append(int(match.group(1)))
            losses.append(float(match.group(2)))

    if not losses:
        raise ValueError(f"No loss values found in {log_path}")

    return steps, losses


def plot_loss(log_path: Path, out_path: Path):
    steps, losses = parse_loss(log_path)

    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, linewidth=1.5)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved loss curve to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot loss curve from MiniGram train log.")
    parser.add_argument("log", nargs="?", default="train_log", help="Path to train log.")
    parser.add_argument("-o", "--output", default="loss_curve.png", help="Output image path.")
    args = parser.parse_args()

    plot_loss(Path(args.log), Path(args.output))


if __name__ == "__main__":
    main()
