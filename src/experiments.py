# Rian - Structured experiments comparing MiDaS vs Depth Anything V2
# Generates comparison grids, edge analysis, indoor vs outdoor analysis,
# and a data table with numerical backing

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import os
from depth_estimator import load_model as load_midas, load_transforms, estimate_depth as midas_depth
from depth_anything import load_model as load_dav2, estimate_depth as dav2_depth

# normalizes depth map to 0-255 for visualization
def normalize_depth(depth):
    d = depth - depth.min()
    d = d / (d.max() + 1e-8)
    return d

# computes edge sharpness score using Laplacian variance
def edge_sharpness(depth):
    depth_uint8 = (normalize_depth(depth) * 255).astype(np.uint8)
    laplacian = cv2.Laplacian(depth_uint8, cv2.CV_64F)
    return laplacian.var()

# computes contrast score as std deviation of normalized depth
def depth_contrast(depth):
    return normalize_depth(depth).std()

# computes depth range as ratio of max to min
def depth_range(depth):
    return depth.max() / (depth.min() + 1e-8)

# figure 1 - full model comparison grid all scenes
def figure1_comparison_grid(scenes, midas_model, midas_transform, dav2_processor, dav2_model):
    print("Generating Figure 1 - full comparison grid...")
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Figure 1 - MiDaS vs Depth Anything V2 across all scenes",
                 fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.08)

    col_titles = ["Original Image", "MiDaS", "Depth Anything V2"]
    row_labels  = ["Outdoor bench", "Acorn Street, Boston", "Indoor living room"]

    for row, (path, label) in enumerate(scenes):
        img_rgb, midas_map = midas_depth(path, midas_model, midas_transform)
        _, dav2_map = dav2_depth(path, dav2_processor, dav2_model)

        for col, (img, cmap) in enumerate([(img_rgb, None),
                                            (midas_map, "inferno"),
                                            (dav2_map,  "inferno")]):
            ax = fig.add_subplot(gs[row, col])
            if cmap:
                ax.imshow(normalize_depth(img), cmap=cmap)
            else:
                ax.imshow(img)
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=13,
                             fontweight="bold", pad=8)
            if col == 0:
                ax.text(-0.08, 0.5, row_labels[row],
                        transform=ax.transAxes,
                        fontsize=11, fontweight="bold", rotation=90,
                        va="center", ha="center", color="#444444")

    plt.savefig("../outputs/fig1_comparison_grid.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig1_comparison_grid.png")

# figure 2 - indoor vs outdoor analysis
def figure2_indoor_outdoor(scenes, midas_model, midas_transform, dav2_processor, dav2_model):
    print("Generating Figure 2 - indoor vs outdoor analysis...")

    # scenes[0] = bench (outdoor), scenes[2] = living room (indoor)
    outdoor_path = scenes[0][0]
    indoor_path  = scenes[2][0]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Figure 2 - Indoor vs Outdoor depth estimation comparison",
                 fontsize=15, fontweight="bold")

    for row, (path, scene_label) in enumerate([
        (outdoor_path, "Outdoor - bench"),
        (indoor_path,  "Indoor - living room"),
    ]):
        img_rgb, midas_map = midas_depth(path, midas_model, midas_transform)
        _, dav2_map = dav2_depth(path, dav2_processor, dav2_model)

        midas_norm = normalize_depth(midas_map)
        dav2_norm  = normalize_depth(dav2_map)
        diff_map   = np.abs(midas_norm - dav2_norm)

        titles = ["Original", "MiDaS depth", "Depth Anything V2", "Difference map"]
        imgs   = [img_rgb, midas_map, dav2_map, diff_map]
        cmaps  = [None, "inferno", "inferno", "RdYlGn_r"]

        for col, (img, title, cmap) in enumerate(zip(imgs, titles, cmaps)):
            ax = axes[row, col]
            if cmap:
                ax.imshow(normalize_depth(img) if col < 3 else img, cmap=cmap)
            else:
                ax.imshow(img)
            ax.axis("off")
            if row == 0:
                ax.set_title(title, fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(scene_label, fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig("../outputs/fig2_indoor_outdoor.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig2_indoor_outdoor.png")

# figure 3 - edge sharpness zoom comparison on boston street
def figure3_edge_analysis(scenes, midas_model, midas_transform, dav2_processor, dav2_model):
    print("Generating Figure 3 - edge sharpness analysis...")

    # scenes[1] = Acorn Street Boston - most dramatic edge difference
    path = scenes[1][0]
    img_rgb, midas_map = midas_depth(path, midas_model, midas_transform)
    _, dav2_map = dav2_depth(path, dav2_processor, dav2_model)

    h, w = midas_map.shape
    # crop center region where lamppost is
    r1, r2 = int(h * 0.1), int(h * 0.9)
    c1, c2 = int(w * 0.3), int(w * 0.7)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Figure 3 - Edge sharpness analysis on Acorn Street, Boston",
                 fontsize=14, fontweight="bold")

    for col, (depth, label) in enumerate([(midas_map, "MiDaS"),
                                           (dav2_map,  "Depth Anything V2")]):
        norm = normalize_depth(depth)
        crop = norm[r1:r2, c1:c2]

        axes[0, col].imshow(norm, cmap="inferno")
        rect = Rectangle((c1, r1), c2 - c1, r2 - r1,
                          linewidth=2, edgecolor="white", facecolor="none")
        axes[0, col].add_patch(rect)
        axes[0, col].set_title(f"{label} - full", fontsize=12, fontweight="bold")
        axes[0, col].axis("off")

        axes[1, col].imshow(crop, cmap="inferno")
        axes[1, col].set_title(f"{label} - zoomed region",
                               fontsize=12, fontweight="bold")
        axes[1, col].axis("off")

    # difference map
    diff = np.abs(normalize_depth(midas_map) - normalize_depth(dav2_map))
    axes[0, 2].imshow(diff, cmap="hot")
    axes[0, 2].set_title("Difference map (full)", fontsize=12, fontweight="bold")
    axes[0, 2].axis("off")

    axes[1, 2].imshow(diff[r1:r2, c1:c2], cmap="hot")
    axes[1, 2].set_title("Difference map (zoomed)", fontsize=12, fontweight="bold")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig("../outputs/fig3_edge_analysis.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig3_edge_analysis.png")

# figure 4 - data table visualization with numbers
def figure4_data_table(scenes, midas_model, midas_transform, dav2_processor, dav2_model):
    print("Generating Figure 4 - data table with numerical analysis...")

    scene_labels = ["Outdoor bench", "Acorn Street", "Indoor room"]
    scene_types  = ["Outdoor", "Outdoor", "Indoor"]

    midas_sharp, dav2_sharp = [], []
    midas_contrast, dav2_contrast = [], []

    for path, label in scenes:
        _, midas_map = midas_depth(path, midas_model, midas_transform)
        _, dav2_map  = dav2_depth(path, dav2_processor, dav2_model)

        midas_sharp.append(round(edge_sharpness(midas_map), 2))
        dav2_sharp.append(round(edge_sharpness(dav2_map), 2))
        midas_contrast.append(round(depth_contrast(midas_map), 4))
        dav2_contrast.append(round(depth_contrast(dav2_map), 4))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 4 - Numerical comparison across scenes",
                 fontsize=14, fontweight="bold")

    x = np.arange(len(scene_labels))
    width = 0.35
    colors = ["#378ADD", "#1D9E75"]

    # edge sharpness
    axes[0].bar(x - width/2, midas_sharp, width, label="MiDaS", color=colors[0])
    axes[0].bar(x + width/2, dav2_sharp,  width, label="Depth Anything V2", color=colors[1])
    axes[0].set_title("Edge sharpness (higher = sharper)", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(scene_labels, rotation=12)
    axes[0].legend()
    axes[0].set_ylabel("Laplacian variance")
    for i, (m, d) in enumerate(zip(midas_sharp, dav2_sharp)):
        axes[0].text(i - width/2, m + 0.5, str(m), ha="center", fontsize=9)
        axes[0].text(i + width/2, d + 0.5, str(d), ha="center", fontsize=9)

    # depth contrast
    axes[1].bar(x - width/2, midas_contrast, width, label="MiDaS", color=colors[0])
    axes[1].bar(x + width/2, dav2_contrast,  width, label="Depth Anything V2", color=colors[1])
    axes[1].set_title("Depth contrast (higher = more variation)", fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(scene_labels, rotation=12)
    axes[1].legend()
    axes[1].set_ylabel("Std deviation")
    for i, (m, d) in enumerate(zip(midas_contrast, dav2_contrast)):
        axes[1].text(i - width/2, m + 0.001, str(m), ha="center", fontsize=9)
        axes[1].text(i + width/2, d + 0.001, str(d), ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("../outputs/fig4_data_table.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # print full table to terminal
    print("\n===== NUMERICAL ANALYSIS TABLE =====")
    print(f"{'Scene':<20} {'Type':<10} {'MiDaS Sharp':>12} {'DAv2 Sharp':>12} {'MiDaS Cont':>12} {'DAv2 Cont':>12}")
    print("-" * 80)
    for i, (sl, st) in enumerate(zip(scene_labels, scene_types)):
        print(f"{sl:<20} {st:<10} {midas_sharp[i]:>12} {dav2_sharp[i]:>12} {midas_contrast[i]:>12} {dav2_contrast[i]:>12}")
    print("=" * 80)

# main function
def main():
    images_dir = "../images"

    # hardcoded in correct order: bench, boston, living room
    scenes = [
        (os.path.join(images_dir, "db6b_bronze_main.jpg"), "Outdoor bench"),
        (os.path.join(images_dir, "test.jpg"), "Acorn Street, Boston"),
        (os.path.join(images_dir, "southernliving-livingroomlayoutmistakes-harperharris-LaureyGlenn-065b1e66d50b4608bf3e74a1f0b0f452.jpg"), "Indoor living room"),
    ]

    print(f"Running experiments on {len(scenes)} scenes")

    print("Loading MiDaS...")
    midas_model = load_midas()
    midas_transform = load_transforms()

    print("Loading Depth Anything V2...")
    dav2_processor, dav2_model = load_dav2()

    figure1_comparison_grid(scenes, midas_model, midas_transform, dav2_processor, dav2_model)
    figure2_indoor_outdoor(scenes, midas_model, midas_transform, dav2_processor, dav2_model)
    figure3_edge_analysis(scenes, midas_model, midas_transform, dav2_processor, dav2_model)
    figure4_data_table(scenes, midas_model, midas_transform, dav2_processor, dav2_model)

    print("\nAll experiment figures saved to outputs/ folder!")
    print("Files: fig1_comparison_grid.png, fig2_indoor_outdoor.png, fig3_edge_analysis.png, fig4_data_table.png")

if __name__ == "__main__":
    main()