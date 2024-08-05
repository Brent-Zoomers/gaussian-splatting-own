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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state, build_rotation
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def cartesian_to_spherical(x, y, z):
    r = torch.sqrt(x**2 + y**2 + z**2)
    phi = torch.acos(z / r)
    theta = torch.atan2(y, x)
    return torch.stack((theta, phi), dim=-1)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_red")
    # gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    # makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.sg_degree + 1)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)


        # Add red color to all splats
        # Define red light
        lambda_ = torch.tensor(1).cuda()
        alpha_ = torch.tensor(10).cuda()
        red_gaussian_position = torch.tensor([-1,3,0]).cuda().float()
        red_gaussian_direction = torch.tensor([0,-1,0]).cuda().float()
        red_gaussian_range = 5
        red_gaussian_half_angle_radians = torch.arccos((torch.log(torch.tensor(0.5))/lambda_)+1.0)
        red_gaussian_direction_normalized = red_gaussian_direction / red_gaussian_direction.norm()

        # For all points calculate if inside cone
        points_ = gaussians.get_xyz
        tensor_to_points = points_-red_gaussian_position 

        dot_product = torch.sum(tensor_to_points * (red_gaussian_direction_normalized), dim=1, keepdim=True)

        distance_on_dir = dot_product / (red_gaussian_direction_normalized).norm()
        # distance_on_dir = dot_product / torch.norm(red_gaussian_direction)

        x = distance_on_dir > 0
        y = distance_on_dir < red_gaussian_range

        # Dot product of unit vectors / norm of apex- point gives angle -> check against half_angle

        norm_dir = red_gaussian_direction / red_gaussian_direction.norm()
        tensor_to_points_norm = tensor_to_points / tensor_to_points.norm(dim=1, keepdim=True)

        dot_product_angle = torch.arccos(torch.sum(tensor_to_points_norm * norm_dir, dim=1, keepdim=True))
    
        mask = (x&y&(torch.abs(dot_product_angle) < red_gaussian_half_angle_radians)).squeeze(-1) #x&y &(torch.abs(dot_product_angle) < red_gaussian_half_angle_radians)

        # Add fat lobe around normal of which alpha depends on distance and angle with light

        # Use 1/max(d, d_min) for light also cos between normal and vector
        # LIGHT
        distance = tensor_to_points.norm(dim=1)
        lighting_attenuation = 1 / torch.pow(torch.maximum(distance, torch.tensor(1)), 2)

        # NORMAL
        _, indices = gaussians.get_scaling.min(dim=1)
        normals = torch.zeros_like(gaussians.get_xyz)
        normals[torch.arange(gaussians.get_xyz.shape[0]), indices] = 1.0

        rotation_mat = build_rotation(gaussians._rotation)
        normals_world = torch.bmm(rotation_mat, normals.unsqueeze(2)).squeeze()

        cosine_factor = torch.sum(normals_world * red_gaussian_direction_normalized, dim=1, keepdim=True)

        new_sg_dirs = cartesian_to_spherical(normals_world[...,0],normals_world[...,1],normals_world[...,2])
        new_sg_lambdas = torch.tensor(0.8, device="cuda")
        new_sg_alphas = lighting_attenuation.unsqueeze(-1) * torch.clamp_min(-cosine_factor, 0) * alpha_
        
        red_gaussian = torch.zeros((gaussians.get_xyz.shape[0], 3, 4))
        red_gaussian[...,0:2] = new_sg_dirs.unsqueeze(1)
        red_gaussian[...,0,2:3] = new_sg_alphas
        red_gaussian[...,3:4] = new_sg_lambdas
 
        red_gaussian = red_gaussian.cuda().transpose(1,2)

        gaussians._features_sg_dc = torch.cat((gaussians._features_sg_dc , red_gaussian), dim=1)

        # Create actual light only for gaussians inside the cone


        # gaussians._xyz = gaussians._xyz[mask]
        # gaussians._opacity = gaussians._opacity[mask]
        # gaussians._scaling = gaussians._scaling[mask]
        # gaussians._rotation = gaussians._rotation[mask]
        # gaussians._features_dc = gaussians._features_dc[mask]
        # gaussians._features_rest = gaussians._features_rest[mask]
        # gaussians._features_sg_dc = gaussians._features_sg_dc[mask]
        # gaussians._features_sg_rest = gaussians._features_sg_rest[mask]


        # red_gaussian = torch.zeros((1, 3, 4))
        # red_gaussian[...,0:2] = torch.tensor([1,torch.pi/2])
        # red_gaussian[...,0,2:] = torch.tensor([0.5, 0])
        # red_gaussian[...,1:,2:] = torch.tensor([0, 1])
 
        # red_gaussian = red_gaussian.cuda().repeat(gaussians.get_xyz.shape[0],1,1).transpose(1,2)

        # gaussians._features_sg_dc = torch.cat((gaussians._features_sg_dc , red_gaussian), dim=1)
        # gaussians._features_sg_rest[:,2,1:] = -5
        # gaussians._features_sg_dc[:,2,1:] = -5


        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

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