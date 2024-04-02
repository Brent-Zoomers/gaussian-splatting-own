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
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        # Remove Smallest Gaussians 

        scalings = scene.gaussians.get_scaling

        sum_scalings = torch.prod(scalings, dim=1, keepdim=True).squeeze(1)

        _, indices = torch.topk(sum_scalings, k= int(sum_scalings.shape[0] * am), largest=True)

        mask = torch.zeros_like(sum_scalings, dtype=torch.bool).cuda()
        mask[indices] = True

        # Remove the elements corresponding to the indices using boolean indexing
        gaussians._opacity =  gaussians._opacity[mask]
        gaussians._xyz = gaussians._xyz[mask]
        gaussians._scaling = gaussians._scaling[mask]
        gaussians._features_dc = gaussians._features_dc[mask]
        gaussians._features_rest = gaussians._features_rest[mask]
        gaussians._rotation = gaussians._rotation [mask]
        #

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, am)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, am)

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

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.amount_biggest_entry)

    print(np.mean(np.array(timings)))