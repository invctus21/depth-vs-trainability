"""
Make figures from a results file.

Usage:
    python plot.py results/sigmoid_lr1.0.json
"""
import argparse
import json
import os

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("results_file")
parser.add_argument("--out-dir", default="figures")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
name = os.path.splitext(os.path.basename(args.results_file))[0]

with open(args.results_file) as fh:
    results = json.load(fh)

depths = sorted(set(r["depth"] for r in results))
n_seeds = len(set(r["seed"] for r in results))
max_epochs = results[0].get("max_epochs", 50000)
stop_loss = results[0].get("stop_loss", 0.001)
test_baseline = results[0]["baseline"]
train_baseline = results[0].get("train_baseline", test_baseline)
colors = {d: plt.cm.viridis(i / max(1, len(depths) - 1)) for i, d in enumerate(depths)}


def label(d):
    return f"{d} hidden layer{'s' if d != 1 else ''}"


def jitter(r):
    # nudge points sideways by seed so they don't overlap
    return r["depth"] + (r["seed"] - (n_seeds - 1) / 2) * 0.08


def save(fig_name):
    path = os.path.join(args.out_dir, f"{name}_{fig_name}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print("saved", path)


# Fig 1: training loss vs epoch
plt.figure(figsize=(7, 5))
for r in results:
    xs = [10 * i + 1 for i in range(len(r["history"]))]     # history holds every 10th epoch
    plt.plot(xs, r["history"], color=colors[r["depth"]], alpha=0.6, linewidth=1)
for d in depths:
    plt.plot([], [], color=colors[d], label=label(d))
plt.axhline(train_baseline, linestyle="--", color="gray", label="predict-the-mean level (train)")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Training loss (MSE, log scale)")
plt.title("Training loss by depth (each line = one seed)")
plt.legend()
plt.grid(True, alpha=0.3)
save("loss_curves")

# Fig 2: epochs needed to reach the stop loss
plt.figure(figsize=(7, 5))
for r in results:
    if r["stop_epoch"] is None:
        plt.scatter(jitter(r), max_epochs, marker="x", color="red")
    else:
        plt.scatter(jitter(r), r["stop_epoch"], color="tab:blue")
plt.scatter([], [], color="tab:blue", label="reached threshold")
plt.scatter([], [], marker="x", color="red", label=f"never reached ({max_epochs:,} epoch limit)")
plt.xticks(depths)
plt.yscale("log")
plt.xlabel("Hidden layers")
plt.ylabel(f"Epochs to reach training loss < {stop_loss}")
plt.title("Time to learn vs depth")
plt.legend()
plt.grid(True, alpha=0.3)
save("epochs_vs_depth")

# Fig 3: gradient size per layer at the first logged snapshot (epoch 100), seed 0
plt.figure(figsize=(7, 5))
for d in depths:
    run = next((r for r in results if r["depth"] == d and r["seed"] == 0), None)
    if run is None or not run["grad_history"]:
        continue                                            # run ended before epoch 100
    norms = run["grad_history"][0][::-1]                    # output layer first
    points = [(i, v) for i, v in enumerate(norms) if v > 0] # zeros cannot be drawn on a log axis
    if points:
        xs, ys = zip(*points)
        plt.plot(xs, ys, marker="o", color=colors[d], label=label(d))
plt.yscale("log")
plt.xlabel("Layers back from the output (0 = output layer)")
plt.ylabel("Gradient size (log scale)")
plt.title("Gradient size per layer at epoch 100 (seed 0)")
plt.legend()
plt.grid(True, alpha=0.3)
save("gradients")

# Fig 4: test loss vs depth
plt.figure(figsize=(7, 5))
for r in results:
    plt.scatter(jitter(r), r["test_loss"], color=colors[r["depth"]])
plt.axhline(test_baseline, linestyle="--", color="red", label="predict-the-mean baseline (test)")
plt.xticks(depths)
plt.yscale("log")
plt.xlabel("Hidden layers")
plt.ylabel("Test loss (MSE, log scale)")
plt.title("Test loss by depth")
plt.legend()
plt.grid(True, alpha=0.3)
save("test_loss")