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
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import copy

""""
@input : Nx3
@output : Nx2

"""
def cartesian2polar(input_tensors):
    real_eigenvectors = torch.real(input_tensors)

    x = real_eigenvectors[...,0]
    y = real_eigenvectors[...,1]
    z = real_eigenvectors[...,2]
    r = torch.sqrt(x**2 + y**2 + z**2)

    # r = torch.sqrt(x**2 + y**2 + z**2) is one
    theta = torch.atan2(y, x)
    phi = torch.acos(z/r)

    polar_coordinates = torch.stack((phi.unsqueeze(-1), theta.unsqueeze(-1)), dim=-1).squeeze(-1)

    return polar_coordinates

""""
@input : Nx2
@output : Nx3

"""
def polar2cartesian(input_tensors):
    phi = input_tensors[...,0]
    theta = input_tensors[...,1]

    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)

    return torch.stack((x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)), dim=-1).squeeze(-1)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(f'output/debug/', name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(f'output/debug/', name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_indices(scene, gaussians, dataset, indices):
    
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    mask = indices.squeeze()


    clone_opacity = gaussians._opacity.clone()
    clone_xyz = gaussians._xyz.clone()
    clone_scaling = gaussians._scaling.clone()
    clone_features_dc = gaussians._features_dc.clone()
    clone_features_rest = gaussians._features_rest.clone()
    clone_rotation = gaussians._rotation.clone()

    gaussians._opacity = gaussians._opacity[mask] 
    gaussians._xyz = gaussians._xyz[mask]
    gaussians._scaling = gaussians._scaling[mask]
    gaussians._features_dc = gaussians._features_dc[mask]
    gaussians._features_rest = gaussians._features_rest[mask]
    gaussians._rotation = gaussians._rotation[mask]

    render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

    gaussians._opacity = clone_opacity
    gaussians._xyz = clone_xyz
    gaussians._scaling = clone_scaling 
    gaussians._features_dc = clone_features_dc
    gaussians._features_rest = clone_features_rest
    gaussians._rotation = clone_rotation






def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    #debug

    # x = torch.tensor([[1,2],[2,1]], dtype=torch.float32)
    # v, vec = torch.linalg.eig(x)

    # print(v, vec)

    # print(np.linalg.eig(x.numpy()))

    #
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # ac_cov = gaussians.get_actual_covariance()

        # values, vectors = torch.linalg.eig(ac_cov[50000])

        # real_values, real_vectors = torch.real(values), torch.real(vectors)
        # real_scaled_vectors = real_values.unsqueeze(1) * real_vectors

        points = gaussians.get_xyz
        actual_covs = gaussians.get_actual_covariance()
        for point in points:
            point = points[800000]
            # calculate distance to all other points
            distances = torch.cdist(point.unsqueeze(0), points)
            values, indices = torch.topk(distances, 5000, largest=False)

            # render_indices(scene, gaussians, dataset, indices.squeeze())

            actual_covs_indexed = actual_covs[indices[0]]
            eigenvalues, eigenvectors = torch.linalg.eig(actual_covs_indexed)

            real_eigenvectors = torch.real(eigenvectors)

            # 0 to only take largest eigenvector into consideration
            polar_coordinates = cartesian2polar(real_eigenvectors[:,0,:])

            phi = polar_coordinates[...,0]
            theta = polar_coordinates[...,1]

            AMOUNT_BUCKETS = 72

            phi_bucketized = torch.bucketize(phi.flatten(), torch.linspace(0, torch.pi, AMOUNT_BUCKETS).cuda())
            theta_bucketized = torch.bucketize(theta.flatten(), torch.linspace(-torch.pi, torch.pi, AMOUNT_BUCKETS).cuda())

            bin_indices = theta_bucketized * AMOUNT_BUCKETS + phi_bucketized

            histogram = torch.bincount(bin_indices, minlength=AMOUNT_BUCKETS * AMOUNT_BUCKETS)

            max_idx = torch.argmax(histogram)

            mask = bin_indices == max_idx

            # only render splats that are in this bin
            # render_indices(scene, gaussians, dataset, indices.squeeze()[mask])
            
            # Find possible features
            # Mask of main directions
            interesting_mask = torch.nonzero(histogram >= histogram[max_idx]*0.8)

            # Convert directions back to xyz
            
            phi = interesting_mask % AMOUNT_BUCKETS
            theta = (interesting_mask-phi) / AMOUNT_BUCKETS

            interesting_polar_cors = torch.stack((phi, theta)).squeeze(-1)

            interesting_cartesian_cors = polar2cartesian(interesting_polar_cors)

            gaussians.get_features[:,0,:]

            feature_vector = interesting_cartesian_cors.flatten()


            # torch.bincount(phi_bucketized)
            # torch.bincount(theta_bucketized)



        print(f'DEBUG')


       

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