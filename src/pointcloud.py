# Rian - Point Cloud Visualization using Open3D
# Takes a depth map and orginal image and creates a 3d point cloud 

import open3d as o3d
import numpy as np
import cv2 
import torch
import matplotlib.pyplot as plt
import os 
from depth_anything import load_model, estimate_depth

# creates a point cloud from an rgb image and depth map
def create_pointcloud(img_rgb, depth_map):
    h, w = depth_map.shape

    # normalizing depth map to 0-1 range 
    depth_normalized = (depth_map-depth_map.min()) / (depth_map.max()- depth_map.min())

    # camera intrensics - standard camera values 
    fx = fy = w
    cx, cy = w/2, h/2

    # creating mesh grid of pixel coordinates 
    u = np.arange(w)
    v = np.arange(h)
    uu, vv = np.meshgrid(u, v)

    # back project pixel to 3D points 
    z = depth_normalized 
    x = (uu-cx) * z / fx
    y = (vv - cy) * z / fy
    
    # stack into N*3 array of 3D points 
    points = np.stack([x, y, z], axis=-1).reshape(-1,3)

    # get colors from orginal image and normalize to 0-1
    colors = img_rgb.reshape(-1, 3) / 255.0

    # create open3d point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

# saves the point cloud to a file 
def save_pointcloud(pcd, output_path):
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Saved point cloud to {output_path}") 

# visualize the point cloud 
def visualize_pointcloud(pcd, title="Point Cloud"):
    print(f"Opening 3d viewer for : {title}")
    print("Use mouse to rotate, scroll to zoom, close window to continue")
    o3d.visualization.draw_geometries([pcd], window_name=title)

# main
def main():
    # paths
    images_dir="../images"
    outputs_dir = "../outputs"

    # load depth anything v2 
    processor, model = load_model()

    # process each image
    for filename in os.listdir(images_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(images_dir, filename)
            pcd_output = os.path.join(outputs_dir, f"pcd_{filename}.ply")

            print(f"Processing {filename}......")

            # get depth map
            img_rgb, depth_map = estimate_depth(image_path, processor, model)

            # create point cloud
            pcd = create_pointcloud(img_rgb, depth_map)

            # save and visualize
            save_pointcloud(pcd, pcd_output)
            visualize_pointcloud(pcd, title=filename)

    print("All point clouds generated!")

if __name__ == "__main__":
    main()