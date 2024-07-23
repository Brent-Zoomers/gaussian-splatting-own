

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

# https://towardsdatascience.com/a-comprehensive-overview-of-gaussian-splatting-e7d570081362
def evaluate_3d_gaussian(point, mean,opacity, covariance):
    # Mahalanobis Distance: torch.mm((point-mean).unsqueeze(0), torch.mm(torch.linalg.inv(covariance), (point-mean).unsqueeze(1)))
    # 0.5 serves as scaling factor to ensure pdf is one
    # exp creates gaussian stuff and opacity scaling factor to put max to opacity
    exponent = -0.5 *  torch.mm((point-mean).unsqueeze(0), torch.mm(torch.linalg.inv(covariance), (point-mean).unsqueeze(1)))
    return opacity * torch.exp(exponent)


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

    mean = gaussians.get_xyz[0]
    opacity = gaussians.get_opacity[0]
    covariance = gaussians.get_actual_covariance()[0]
    point = mean - 0.1

    evaluate_3d_gaussian(point,mean,opacity,covariance)
   

    
   

# Printing the octree structure (optional)




    # with torch.no_grad():
    #     # Generate grid-structure
    #     structure = generate_structure(gaussians)