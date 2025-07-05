import argparse
import re
import sys
import matplotlib.pyplot as plt
import numpy as np 

def parse_log(logfile):
    """
    Parse logfile for epoch, kidney and tumor dice scores.
    Returns three lists: epochs, kidney_scores, tumor_scores.
    """
    pattern = re.compile(
        r"Epoch:\s*(\d+).*?Kidney Dice score:\s*([0-9.]+).*?Tumor Dice score:\s*([0-9.]+)"
    )
    epochs = []
    kidney = []
    tumor = []
    with open(logfile, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append(int(m.group(1)))
                kidney.append(float(m.group(2)))
                tumor.append(float(m.group(3)))
    if not epochs:
        sys.exit(f"No valid 'Kidney Dice score' lines found in {logfile}")
    return epochs, kidney, tumor

def plot_scores(epochs, kidney, tumor, output):
    """
    Plot the kidney and tumor dice scores over epochs.
    If output ends with .png/.pdf/etc, saves to file; otherwise shows on screen.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, kidney, marker='o', label='Kidney Dice')
    plt.plot(epochs, tumor,  marker='s', label='Tumor Dice')
    plt.title('Dice Scores per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # ** set y-axis from 0 to 1 with 0.1 intervals **
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.01, 0.1))

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=300)
        print(f"Saved plot to {output}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Plot kidney & tumor dice scores from a training log."
    )
    parser.add_argument(
        '--logfile', '-l',
        required=True,
        help="Path to the training log file."
    )
    parser.add_argument(
        '--output', '-o',
        default='',
        help="Output image file (e.g. dice.png). If omitted, displays on-screen."
    )
    args = parser.parse_args()

    epochs, kidney, tumor = parse_log(args.logfile)
    plot_scores(epochs, kidney, tumor, args.output)

if __name__ == '__main__':
    main()