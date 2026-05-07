#Rian 

import torch 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

#Loads the Depth Anything V2 model from hugging face 

def load_model():
    print("Loading Depth Anything V2 model.....")
    processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model.eval()
    print("Model loaded successfully!!!")
    return processor, model

# runs depth estimation on a single image file 
def estimate_depth(image_path, processor, model):
    #read the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #Prepare image for model
    inputs = processor(images=img_rgb, return_tensors="pt")

    #run through model
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

        #resize output to match orginal image size 
    depth_map = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = depth_map.numpy()

    return img_rgb, depth_map

#saves and displays the depth map side by side with orginal 

def visualize_depth(img_rgb, depth_map, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(depth_map, cmap="inferno")
    axes[1].set_title("Depth Anything v2 - Depth Map")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"Saved output to {output_path}")


# Main function
def main():
    #paths
    images_dir = "../images"
    outputs_dir = "../outputs"

    # load model and processor
    processor, model = load_model()

    #looping through every image in the images folder
    for filename in os.listdir(images_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(images_dir, filename)
            output_path = os.path.join(outputs_dir, f"dav2_{filename}")

            print(f"Processing {filename}...")
            img_rgb, depth_map = estimate_depth(image_path, processor, model)
            visualize_depth(img_rgb, depth_map, output_path)
    print("All images processed!")

if __name__ =="__main__":
    main()

