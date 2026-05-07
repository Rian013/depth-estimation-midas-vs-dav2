# Rian - Depth Estimator using MiDaS
# Loads a pretrained MiDaS model and runs depth estimation on an input image

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# loads the MiDaS model from torch hub
def load_model():
    print("Loading MiDaS model...")
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    model.eval() #Turns off training behaviour
    print("Model loaded successfully!")
    return model

# loads the correct image transforms for MiDaS
def load_transforms():
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    return transforms.small_transform

# runs depth estimation on a single image file
def estimate_depth(image_path, model, transform):
    # read the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # apply transforms and run through model
    input_batch = transform(img_rgb)
    with torch.no_grad():
        prediction = model(input_batch)

    # resize output to match original image size
    depth_map = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth_map = depth_map.numpy()
    return img_rgb, depth_map

# saves and displays the depth map side by side with original
def visualize_depth(img_rgb, depth_map, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(depth_map, cmap="inferno")
    axes[1].set_title("Depth Map")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"Saved output to {output_path}")

# main function
def main():
    # paths
    images_dir = "../images"
    outputs_dir = "../outputs"

    # load model and transforms
    model = load_model()
    transform = load_transforms()

    #Loop through every image in the image folder 
    for filename in os.listdir(images_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(images_dir, filename)
            output_path = os.path.join(outputs_dir, f"depth_{filename}")

            print(f"Processing {filename}...")
            img_rgb, depth_map = estimate_depth(image_path, model, transform)
            visualize_depth(img_rgb, depth_map, output_path)
    print("All images are porcessed!")

if __name__ == "__main__":
    main()