import torch
import numpy as np
import open3d as o3d

# Step 1: Convert PyTorch tensor to NumPy array
# Example tensor of shape (N, 3) where N is the number of points
points_tensor = torch.rand((5000,3))

# Convert to NumPy array
points_numpy = points_tensor.numpy()

# Step 2: Create an Open3D point cloud from the NumPy array
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points_numpy)

# Step 3: Create an octree from the point cloud
# Define the depth of the octree
octree = o3d.geometry.Octree(max_depth=8)
octree.convert_from_point_cloud(point_cloud, size_expand=0.01)

# Visualization (optional)
o3d.visualization.draw_geometries([octree])


# Printing the octree structure (optional)
print(octree)