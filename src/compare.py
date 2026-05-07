# Rian - Side by side comparison of MiDaS vs Depth Anything V2
# Generates a single figure with original, MiDaS depth, and DAv2 depth per image

import matplotlib.pyplot as plt
import os
from depth_estimator import load_model as load_midas, load_transforms, estimate_depth as midas_depth
from depth_anything import load_model as load_dav2, estimate_depth as dav2_depth

# generates a side by side comparison figure for a single image
def compare_models(image_path, filename, midas_model, midas_transform, dav2_processor, dav2_model, output_path):
    # run both models
    img_rgb, midas_map = midas_depth(image_path, midas_model, midas_transform)
    _, dav2_map = dav2_depth(image_path, dav2_processor, dav2_model)

    # create figure with 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(midas_map, cmap="inferno")
    axes[1].set_title("MiDaS Depth", fontsize=13)
    axes[1].axis("off")

    axes[2].imshow(dav2_map, cmap="inferno")
    axes[2].set_title("Depth Anything V2", fontsize=13)
    axes[2].axis("off")

    plt.suptitle(f"Depth Estimation Comparison: {filename}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved comparison to {output_path}")

# main function
def main():
    images_dir = "../images"
    outputs_dir = "../outputs"

    # load both models once
    print("Loading MiDaS...")
    midas_model = load_midas()
    midas_transform = load_transforms()

    print("Loading Depth Anything V2...")
    dav2_processor, dav2_model = load_dav2()

    # run comparison on every image
    for filename in os.listdir(images_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(images_dir, filename)
            output_path = os.path.join(outputs_dir, f"compare_{filename}")

            print(f"Comparing {filename}...")
            compare_models(image_path, filename, midas_model, midas_transform, dav2_processor, dav2_model, output_path)

    print("All comparisons done!")

if __name__ == "__main__":
    main()