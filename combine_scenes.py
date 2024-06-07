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

timings = []

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, am):
    render_path = os.path.join(model_path, name, "show_largest_{}_{}".format(am, iteration), "renders")
    gts_path = os.path.join(model_path, name, "show_largest_{}_{}".format(am, iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        start_time = time.time()
        rendering = render(view, gaussians, pipeline, background)["render"]
        end_time = time.time()
        elapsed_time_ms = (end_time - start_time) * 1000
        timings.append(elapsed_time_ms)

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,am=0):
    with torch.no_grad():
        dataset.model_path = f'output/random1'
        gaussians1 = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians1, load_iteration=iteration, shuffle=False)

        dataset.model_path = f'output/random2'
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # Remove the elements corresponding to the indices using boolean indexing
        gaussians1._opacity =  torch.cat((gaussians1._opacity, gaussians._opacity), dim=0)
        gaussians1._xyz = torch.cat((gaussians1._xyz, gaussians._xyz), dim=0)
        gaussians1._scaling = torch.cat((gaussians1._scaling, gaussians._scaling), dim=0)
        gaussians1._features_dc = torch.cat((gaussians1._features_dc, gaussians._features_dc), dim=0)
        gaussians1._features_rest = torch.cat((gaussians1._features_rest, gaussians._features_rest), dim=0)
        gaussians1._rotation = torch.cat((gaussians1._rotation, gaussians._rotation), dim=0)
        #

        gaussians1.save_ply("output/combined_random/point_cloud/iteration_30000/point_cloud.ply")

        # bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        # background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # if not skip_train:
        #      render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, am)

        # if not skip_test:
        #      render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, am)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--amount_biggest_entry", type=float)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)

    print(np.mean(np.array(timings)))