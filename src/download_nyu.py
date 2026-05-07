# Rian - Download NYU Depth V2 subset for benchmarking
# Downloads directly from the official NYU source

import urllib.request
import numpy as np
import os
from PIL import Image
import h5py

def main():
    print("Extracting NYU Depth V2 samples...")
    os.makedirs("../data/nyu", exist_ok=True)

    mat_path = "../data/nyu/nyu_depth_v2_labeled.mat"

    # check file exists
    if not os.path.exists(mat_path):
        print("Mat file not found! Please download it first")
        return

    # load using h5py for matlab v7.3 files
    print("Loading mat file...")
    with h5py.File(mat_path, "r") as f:
        images = np.array(f["images"])
        depths = np.array(f["depths"])

    print(f"Dataset loaded! {images.shape[0]} total samples")

    # save first 50 samples
    for i in range(50):
        # images are stored as (N, C, H, W) so transpose to (H, W, C)
        img = images[i].transpose(1, 2, 0)
        img = Image.fromarray(img.astype(np.uint8))
        img.save(f"../data/nyu/rgb_{i:04d}.png")

        # depths are stored as (N, H, W)
        depth = depths[i]
        np.save(f"../data/nyu/depth_{i:04d}.npy", depth)

        if i % 10 == 0:
            print(f"Saved {i+1}/50 samples...")

    print("Done! Data saved to ../data/nyu/")

if __name__ == "__main__":
    main()