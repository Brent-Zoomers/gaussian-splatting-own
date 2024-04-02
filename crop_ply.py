import plyfile
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import numpy as np

def crop_ply(ply_data, min_x, min_y, min_z, max_x, max_y, max_z):
  """Crops a PLY data structure based on a bounding box."""
  vertices = [vertex for vertex in ply_data['vertex'] if (
      min_x <= vertex[0] <= max_x and
      min_y <= vertex[1] <= max_y and
      min_z <= vertex[2] <= max_z
  )]
  faces = [face for face in ply_data['face']]
  return {'vertex': vertices, 'face': faces}

def read_and_crop_ply(filename):
  """Reads a PLY file."""
  ply_data = plyfile.PlyData.read(filename)
  return ply_data

def visualize_ply(ply_data):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(ply_data['vertex']))
    o3d.visualization.draw_geometries([pcd])
 

# Loop through multiple PLY files
filenames = ["C:/Users/bzoomers/Desktop/gaussian-splatting-own/eval/full_eval_0_0/bicycle/point_cloud/iteration_30000/point_cloud.ply"]

for filename in filenames:
  print(f"Processing: {filename}")
  ply_data = read_and_crop_ply(filename)

  # Visualize the point cloud before prompting
  visualize_ply(ply_data)  # Avoid modifying original data

  # Get user input for cropping box with error handling
  while True:
    try:
      min_x = float(input("Enter minimum X (or 'q' to quit): "))
      if min_x == 'q':
        break  # Exit loop on 'q'
      min_y = float(input("Enter minimum Y: "))
      min_z = float(input("Enter minimum Z: "))
      max_x = float(input("Enter maximum X: "))
      max_y = float(input("Enter maximum Y: "))
      max_z = float(input("Enter maximum Z: "))
      break  # Exit loop on successful input
    except ValueError:
      print("Invalid input. Please enter numbers.")

  # Perform cropping based on user input
  cropped_data = crop_ply(ply_data.copy(), min_x, min_y, min_z, max_x, max_y, max_z)

  # ... (your processing code here for the cropped data)



  