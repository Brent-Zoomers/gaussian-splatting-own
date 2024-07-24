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
from utils.general_utils import safe_state, calculate_real_eigenvectors, normal_to_rgb, build_rotation
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    normals_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normals")

    makedirs(render_path, exist_ok=True)
    makedirs(normals_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]

        _, indices = gaussians.get_scaling.min(dim=1)
        normals = torch.zeros_like(gaussians.get_xyz)
        normals[torch.arange(gaussians.get_xyz.shape[0]), indices] = 1.0

        rotation_mat = build_rotation(gaussians._rotation)
        normals_world = torch.bmm(rotation_mat, normals.unsqueeze(2)).squeeze()

        # actual_covariances = gaussians.get_actual_covariance()
        # eigvals, eigvecs = calculate_real_eigenvectors(actual_covariances)
        # _, lowest_indices = torch.min(eigvals, dim=1, keepdim=True)

        # expanded_indices = lowest_indices.expand(-1, 3).unsqueeze(1)
        # smallest_eigenvecs = eigvecs.gather(1, lowest_indices.expand(-1, 3).unsqueeze(1)).squeeze()

        # Gathering the smallest eigenvectors corresponding to the smallest eigenvalues
        # Ensure eigvecs and indices are correctly shaped and contiguous
        w2c = view.world_view_transform.clone()
        w = torch.ones(normals_world.shape[0], 1).cuda()

        homogemous_vecs_world = torch.cat((normals_world, w), dim=1)
        w2c[3,0:3] = 0.0
        vec_end_cam = torch.matmul(homogemous_vecs_world, w2c)


        colors = normal_to_rgb(vec_end_cam[...,:3])


        # Calculate normals using opacity as 1?
        normals = render(view, gaussians, pipeline, background, override_color=colors)["render"]
        # eigen vecs normalizeren
        # naar juiste coordinaatsysteem zetten
        
        # reorder_color = torch.stack((smallest_eigenvecs[...,2],smallest_eigenvecs[...,1],smallest_eigenvecs[...,0]), dim=1)
        # colors = torch.clamp(normalized_eigvecs_cam, 0, 1)

        # gaussians._opacity[:] = 10.0
        # Calculate normals using opacity as 1?
        
        
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(normals, os.path.join(normals_path, '{0:05d}'.format(idx) + ".png"))
        

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

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