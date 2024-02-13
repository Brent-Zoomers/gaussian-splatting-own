#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
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
import time

# FROM forward.cu
# // Spherical harmonics coefficients
# __device__ const float SH_C0 = 0.28209479177387814f;
# __device__ const float SH_C1 = 0.4886025119029199f;
# __device__ const float SH_C2[] = {
# 	1.0925484305920792f,
# 	-1.0925484305920792f,
# 	0.31539156525252005f,
# 	-1.0925484305920792f,
# 	0.5462742152960396f
# };
# __device__ const float SH_C3[] = {
# 	-0.5900435899266435f,
# 	2.890611442640554f,
# 	-0.4570457994644658f,
# 	0.3731763325901154f,
# 	-0.4570457994644658f,
# 	1.445305721320277f,
# 	-0.5900435899266435f
# };


def generate_uniform_rays(num_rays):
    # Generate random values for theta and phi
    theta = torch.acos(1 - 2 * torch.rand(num_rays))  # Polar angle
    phi = 2 * torch.pi * torch.rand(num_rays)        # Azimuthal angle

    # Convert spherical coordinates to Cartesian coordinates
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)

    # Stack the coordinates to form the rays
    rays = torch.stack((x, y, z), dim=-1)

    # normalized_rays = F.normalize(rays, p=2, dim=-1)

    return rays


def calc_sh(weights, direction):
    # band
    # 0
    # 1
    # 2
    # 3

    # 16
    coefficients = torch.tensor([
        0.28209479177387814,
        0.4886025119029199,0.4886025119029199,0.4886025119029199,
        1.0925484305920792,	-1.0925484305920792, 0.31539156525252005, -1.0925484305920792,0.5462742152960396,
        -0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435
        ])
    # 16x1
    coefficients = coefficients.unsqueeze(1)

    # N x 16 x 3
    constants = weights * coefficients.cuda() 

    # S
    x, y, z = direction.unbind(1)

    # N x S x 3 (per gaussians colors in all directions)
    result = torch.zeros((weights.shape[0], x.shape[0], 3)).cuda() # N x S

    # band 0
    result += constants[:,0,:]

    #band 1
    result += constants[:,1,:] * y
    result += constants[:,2,:] * z
    result += constants[:,3,:] * x

    #band 2
    result += constants[:,4,:] * x*y
    result += constants[:,5,:] * x*z
    result += constants[:,6,:] * 3*(z*z)-1
    result += constants[:,7,:] * x*z
    result += constants[:,8,:] * x*x-y*y

    #band 3
    result += constants[:,9,:] * x*y
    result += constants[:,10,:] * x*z
    result += constants[:,11,:] * 3*(z*z)-1
    result += constants[:,12,:] * x*z
    result += constants[:,13,:] * x*x-y*y
    result += constants[:,14,:] * x*x-y*y
    result += constants[:,15,:] * x*x-y*y

    return result

timings = []

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "reduced_{}_{}".format(am, iteration), "renders")
    gts_path = os.path.join(model_path, name, "reduced_{}_{}".format(am, iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        start_time = time.time()
        rendering = render(view, gaussians, pipeline, background)["render"]
        end_time = time.time()
        elapsed_time_ms = (end_time - start_time) * 1000
        timings.append(elapsed_time_ms)

        gt = view.original_image[0:3, :, :]
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # Remove Smallest Gaussians 

       
        # Remove the elements corresponding to the indices using boolean indexing
        # gaussians._opacity =  gaussians._opacity[mask]
        # gaussians._xyz = gaussians._xyz[mask]
        # gaussians._scaling = gaussians._scaling[mask]
        # gaussians._features_dc = gaussians._features_dc[mask]
        # gaussians._features_rest = gaussians._features_rest[mask]
        # gaussians._rotation = gaussians._rotation [mask]
        #

        # calc sh_color in direction
        rays = generate_uniform_rays(100)
        calc_sh(gaussians.get_features, rays)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

       

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

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)

    print(np.mean(np.array(timings)))