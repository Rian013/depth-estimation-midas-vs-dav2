# Rian - Benchmark script for MiDaS vs Depth Anything V2
# Evaluates both models on NYU Depth V2 using AbsRel and RMSE metrics

import numpy as np
import os
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from depth_estimator import load_model as load_midas, load_transforms, estimate_depth as midas_depth
from depth_anything import load_model as load_dav2, estimate_depth as dav2_depth

# computes absolute relative error between predicted and ground truth depth
def abs_rel(pred, gt):
    return np.mean(np.abs(pred - gt) / gt)

# computes root mean squared error between predicted and ground truth depth
def rmse(pred, gt):
    return np.sqrt(np.mean((pred - gt) ** 2))

# aligns predicted depth scale to ground truth using least squares
def align_depth(pred, gt):
    # least squares scale and shift alignment
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()
    scale = np.dot(pred_flat, gt_flat) / np.dot(pred_flat, pred_flat)
    return pred * scale

# runs evaluation on all NYU samples
def evaluate(model_name, estimate_fn, model_args, data_dir):
    abs_rel_scores = []
    rmse_scores = []

    samples = sorted([f for f in os.listdir(data_dir) if f.startswith("rgb_")])
    print(f"Evaluating {model_name} on {len(samples)} samples...")

    for filename in samples:
        idx = filename.split("_")[1].split(".")[0]
        rgb_path = os.path.join(data_dir, filename)
        depth_path = os.path.join(data_dir, f"depth_{idx}.npy")

        # load ground truth depth
        gt_depth = np.load(depth_path)

        # get predicted depth
        img_rgb, pred_depth = estimate_fn(rgb_path, *model_args)

        # resize pred to match gt size
        pred_resized = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]))

        # align scale
        pred_aligned = align_depth(pred_resized, gt_depth)

        # compute metrics
        abs_rel_scores.append(abs_rel(pred_aligned, gt_depth))
        rmse_scores.append(rmse(pred_aligned, gt_depth))

    mean_absrel = np.mean(abs_rel_scores)
    mean_rmse = np.mean(rmse_scores)

    print(f"{model_name} - AbsRel: {mean_absrel:.4f}, RMSE: {mean_rmse:.4f}")
    return mean_absrel, mean_rmse

# plots a bar chart comparing both models
def plot_results(results):
    models = list(results.keys())
    absrel_vals = [results[m]["absrel"] for m in models]
    rmse_vals = [results[m]["rmse"] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].bar(models, absrel_vals, color=["steelblue", "coral"])
    axes[0].set_title("AbsRel (lower is better)")
    axes[0].set_ylabel("AbsRel Error")

    axes[1].bar(models, rmse_vals, color=["steelblue", "coral"])
    axes[1].set_title("RMSE (lower is better)")
    axes[1].set_ylabel("RMSE Error")

    plt.tight_layout()
    plt.savefig("../outputs/benchmark_results.png")
    plt.show()
    print("Saved benchmark chart to ../outputs/benchmark_results.png")

# main function
def main():
    data_dir = "../data/nyu"

    # load midas
    print("Loading MiDaS...")
    midas_model = load_midas()
    midas_transform = load_transforms()

    # load depth anything v2
    print("Loading Depth Anything V2...")
    processor, dav2_model = load_dav2()

    # evaluate both models
    results = {}

    absrel, rms = evaluate("MiDaS", midas_depth, [midas_model, midas_transform], data_dir)
    results["MiDaS"] = {"absrel": absrel, "rmse": rms}

    absrel, rms = evaluate("Depth Anything V2", dav2_depth, [processor, dav2_model], data_dir)
    results["Depth Anything V2"] = {"absrel": absrel, "rmse": rms}

    # print final results
    print("\n===== FINAL RESULTS =====")
    for model, scores in results.items():
        print(f"{model}: AbsRel={scores['absrel']:.4f}, RMSE={scores['rmse']:.4f}")

    # plot comparison chart
    plot_results(results)

if __name__ == "__main__":
    main()