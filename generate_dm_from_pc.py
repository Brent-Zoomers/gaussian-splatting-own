import os
import torch
import numpy as np
from PIL import Image
import open3d as o3d

# Define a function to project 3D points to 2D image plane using PyTorch
def project_to_image(points, focal_length_x, focal_length_y, width, height):
    x = (points[:, 0] * focal_length_x) / points[:, 2] + width / 2
    y = (points[:, 1] * focal_length_y) / points[:, 2] + height / 2
    z = points[:, 2]
    return torch.stack((x, y, z), dim=-1)

# Process each point cloud file
for k, filename in enumerate(filenames):
    print(f'Processing {k+1}/{len(filenames)}: {filename}')

    # Load the point cloud
    pcd = o3d.io.read_point_cloud(filename)
    points = torch.tensor(np.asarray(pcd.points), dtype=torch.float32)
    colors = torch.tensor(np.asarray(pcd.colors) * 255, dtype=torch.uint8)  # Convert colors back to 0-255 range

    # Get the image dimensions and focal lengths
    width = args.width
    height = args.height
    focal_length_x = args.focal_length_x
    focal_length_y = args.focal_length_y

    # Project 3D points to 2D image plane
    projected_points = project_to_image(points, focal_length_x, focal_length_y, width, height)
    x, y, z = projected_points[:, 0], projected_points[:, 1], projected_points[:, 2]

    # Create an empty depth map
    depth_map = torch.zeros((height, width), dtype=torch.float32)

    # Fill the depth map with the z-values (depth) from the projected points
    valid_indices = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    depth_map[y[valid_indices].long(), x[valid_indices].long()] = z[valid_indices]

    # Normalize the depth map for visualization (optional)
    min_depth = torch.min(depth_map)
    max_depth = torch.max(depth_map)
    depth_map_normalized = (depth_map - min_depth) / (max_depth - min_depth)
    depth_map_image = (depth_map_normalized * 255).to(torch.uint8).cpu().numpy()

    # Save the depth map as an image file
    depth_map_filename = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + "_depth.png")
    Image.fromarray(depth_map_image).save(depth_map_filename)

    print(f'Saved depth map to {depth_map_filename}')
