

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
            
  
    return stacked
    




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

    with torch.no_grad():
        # Generate grid-structure
        structure = generate_structure(gaussians)