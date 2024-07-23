

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import open3d as o3d
import time
from render import render_indices

def get_average_color(colors):
    return torch.sum(colors, dim=0) / colors.shape[0]

def get_variance_color(colors):
    if colors.shape[0] == 1:
        return colors
    return torch.var(colors, dim=0)

def create_descriptor(point, gaussians):
    pass



def generate_structure(gaussians, levels = 2):
    xyz = gaussians.get_xyz
    colors = gaussians.get_features[:,0,:]

    levels = {}

    # Split point cloud into 8 and hold set of points per voxel
    x_max, x_min = torch.max(xyz[...,0]),torch.min(xyz[...,0])
    y_max, y_min = torch.max(xyz[...,1]),torch.min(xyz[...,1])
    z_max, z_min = torch.max(xyz[...,2]),torch.min(xyz[...,2])

    x_range = torch.linspace(x_min, x_max, 3).cuda()
    x_masked = x_range.unsqueeze(1) >= xyz[...,0].unsqueeze(0).repeat(3,1)
    x_indices = x_range.shape[0] - x_masked.count_nonzero(dim=0)

    y_range = torch.linspace(y_min, y_max, 3).cuda()
    y_masked = y_range.unsqueeze(1) >= xyz[...,1].unsqueeze(0).repeat(3,1)
    y_indices = y_range.shape[0] - y_masked.count_nonzero(dim=0)

    z_range = torch.linspace(z_min, z_max, 3).cuda()
    z_masked = z_range.unsqueeze(1) >= xyz[...,2].unsqueeze(0).repeat(3,1)
    z_indices = z_range.shape[0] - z_masked.count_nonzero(dim=0)

    stacked = torch.stack((x_indices, y_indices, z_indices)).permute(1,0)

    # Find points in each block, calc variance and recursively this function
    unique_entries = torch.unique(stacked, dim=0)

    for entry in unique_entries:
        x = (stacked == entry).all(dim=1)
        indices = torch.nonzero(x).squeeze(1)

        if torch.sum(torch.abs(get_variance_color(colors[indices]))) > 3 :
            pass 
  
    return stacked


def own_traverse(node, parent=None, max_ancestor_variance=0.0, id_string=""):
    filtered_colors = np.asarray(point_cloud.colors)[np.asarray(node.indices)]       
    var_per_channel = np.var(filtered_colors, axis=0)
    variance = np.sum(var_per_channel)

    max_variance_children = 0.0
    if not isinstance(node, o3d.geometry.OctreeLeafNode):
        for idx, child in enumerate(node.children):
            if child is not None:
                highest_var = max(max_ancestor_variance, variance)
                max_variance_children = max(own_traverse(child, node, highest_var, id_string+str(idx)), max_variance_children)

        if variance > max_variance_children and variance > max_ancestor_variance:
            render_indices(scene, gaussians, dataset, pipeline, torch.tensor(np.asarray(node.indices), dtype=torch.long), f"variance_based_20levels/{id_string}/")
    
    
    if isinstance(node, o3d.geometry.OctreeLeafNode):
        if variance > max_ancestor_variance:
            render_indices(scene, gaussians, dataset, pipeline, torch.tensor(np.asarray(node.indices), dtype=torch.long), f"variance_based_20levels/{id_string}/")
    
    return max(variance, max_variance_children)

    

def f_traverse(node, node_info):
    early_stop = False

    if isinstance(node, o3d.geometry.OctreeInternalNode):
        if isinstance(node, o3d.geometry.OctreeInternalPointNode):
            
            # point_cloud.points[node.indices]
            filtered_colors = np.asarray(point_cloud.colors)[np.asarray(node.indices)]       
            var_per_channel = np.var(filtered_colors, axis=0)

            if np.sum(var_per_channel) > 0.25:
                print("{}{}: Internal node at depth {}: var {}"
                      .format('    ' * node_info.depth, node_info.child_index, node_info.depth,np.sum(var_per_channel)))
           
            # DEBUG : print out all cells that have variance of one or more and see if they are interesting
                render_indices(scene, gaussians, dataset, pipeline, torch.tensor(np.asarray(node.indices), dtype=torch.long), f"variance_based/{node_info.depth}_{node_info.child_index}/")

           
            early_stop = len(node.indices) < 250
    elif isinstance(node, o3d.geometry.OctreeLeafNode):
        if isinstance(node, o3d.geometry.OctreePointColorLeafNode):
            pass
            # print("{}{}: Leaf node at depth {} has {} points with origin {}".
                #   format('    ' * node_info.depth, node_info.child_index,
                        #  node_info.depth, len(node.indices), node_info.origin))
    else:
        raise NotImplementedError('Node type not recognized!')

    # early stopping: if True, traversal of children of the current node will be skipped
    return early_stop


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    dataset = model.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    # Convert to NumPy array
    points_numpy = gaussians.get_xyz.cpu().detach().numpy()
    colors_numpy = (gaussians.get_features[:,0,:] * 0.28209479177387814 + 0.5).cpu().detach().numpy()

    # Step 2: Create an Open3D point cloud from the NumPy array
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_numpy)
    point_cloud.colors = o3d.utility.Vector3dVector(colors_numpy)

    # Step 3: Create an octree from the point cloud
    # Define the depth of the octree
    octree = o3d.geometry.Octree(max_depth=20)
    octree.convert_from_point_cloud(point_cloud, size_expand=0.01)

    # Visualization (optional)
    # o3d.visualization.draw_geometries([octree])

    start_time = time.time()
    octree.traverse(f_traverse)
    # own_traverse(octree.root_node)
    end_time = time.time()
    # print(octree.locate_leaf_node(point_cloud.points[0]))
    

    # Calculate the execution time
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.10f} seconds")


