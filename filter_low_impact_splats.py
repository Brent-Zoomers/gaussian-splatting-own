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
import timeit
import shutil

timings = []

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, am):
    render_path = os.path.join(model_path, name, "show_largest_{}_{}".format(am, iteration), "renders")
    gts_path = os.path.join(model_path, name, "show_largest_{}_{}".format(am, iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        print(view.image_name)
        # start_time = time.time()
        rendering = render(view, gaussians, pipeline, background)["render"]
        # end_time = time.time()
        # elapsed_time_ms = (end_time - start_time) * 1000
        # timings.append(elapsed_time_ms)

        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,am=0):
    
    with torch.no_grad():
        # dataset.model_path = f'output/combined_random'
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        start_time = time.time()
        
        # Determine useful/useless splats to be removed

        # Render to each viewpoint and hold list of contributing splats
        # Take N splats per pixel per view and only hold splats that occured M times?

        cameras = scene.getTrainCameras()
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        x = torch.zeros((gaussians.get_xyz.shape[0]), device='cuda')
        total_contr = torch.zeros((gaussians.get_xyz.shape[0]), device='cuda')
        total_occ = torch.zeros((gaussians.get_xyz.shape[0]), device='cuda', dtype=torch.long)

        total_contr_lol = torch.zeros((gaussians.get_xyz.shape[0]), device='cuda')
        total_occ_lol = torch.zeros((gaussians.get_xyz.shape[0]), device='cuda', dtype=torch.long)

        xx = torch.zeros((gaussians.get_xyz.shape[0]), device='cuda')
        # Work using score

        # amount contributed * (max contribution)

        ################################## 
        for camera in cameras:
            # print(camera.image_name)
        #     # Render to camera
            render_pkg = render(camera, gaussians, pipeline, background)
        #     gt_image = camera.original_image.cuda()

        #     num_gaussians_per_pixel = 20

        #     # l1loss_per_pixel = torch.abs(render_pkg["render"] - gt_image).sum(dim=0, keepdim=True).repeat(num_gaussians_per_pixel,1,1).flatten()
            
        #     # for each pixel weight depending on loss/contribution
            ids_per_pixel = render_pkg["ids_per_pixel"]
            contr_per_pixel = render_pkg["contr_per_pixel"]

            per_pixel_reshaped = ids_per_pixel.view((camera.image_width, camera.image_height,20))

            most_contributing_per_pixel = per_pixel_reshaped[...,-am]
        #     #(pix_id*NUM_GAUSSIANS_CONTRIBUTING)+index
        #     # uint32_t pix_id = W * pix.y + pix.x;

            most_per_pixel = torch.unique(most_contributing_per_pixel.flatten())

            y = torch.unique(ids_per_pixel, return_counts=True)
        #     # y[1] *= l1loss_per_pixel
            x[y[0]] += y[1]
            xx[most_per_pixel] += 1


            set = torch.unique(ids_per_pixel)
            total_contr.index_add_(0, ids_per_pixel, contr_per_pixel)
            total_occ.index_add_(0, set, torch.ones_like(set))

            total_contr_lol[ids_per_pixel] += contr_per_pixel
            total_occ_lol[ids_per_pixel] += 1

        ################################## 

        weighted_scale_opacity = -torch.exp(torch.sum(gaussians.get_scaling,dim=1)) / (1 + torch.exp(-gaussians.get_opacity.squeeze()))

        # *total_occ*gaussians.get_opacity.squeeze()
        largest_k_values, largest_k_indices = torch.topk(total_contr, int(total_contr.shape[0]*(am/10.0)), largest=True)
        # largest_k_values, largest_k_indices = torch.topk(weighted_scale_opacity, int(gaussians.get_opacity.shape[0]*am/10.0), largest=False)
        # end_time = time.time()
        # execution_time = end_time - start_time
        # print("Execution time:", execution_time, "seconds")

        
        # mask = xx != 0
        mask = largest_k_indices

        # Remove the elements corresponding to the indices using boolean indexing
        gaussians._opacity = gaussians._opacity[mask] 
        gaussians._xyz = gaussians._xyz[mask]
        gaussians._scaling = gaussians._scaling[mask]
        gaussians._features_dc = gaussians._features_dc[mask]
        gaussians._features_rest = gaussians._features_rest[mask]
        gaussians._rotation = gaussians._rotation[mask]
        #

        dataset_name = dataset.model_path.split("/")[-1]
        output_folder_path = f'output/datasets/{dataset_name}ivan/{am}'
        folder_path = f'{output_folder_path}/point_cloud/iteration_30000'
        if not os.path.exists(folder_path):
                # If the folder does not exist, create it  
                os.makedirs(folder_path)
        gaussians.save_ply(folder_path + '/point_cloud.ply')

        # copy files to right folder

        shutil.copy(dataset.model_path+'/cameras.json', output_folder_path)
        shutil.copy(dataset.model_path+'/cfg_args', output_folder_path)


       
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
    for i in range(1, 6, 1):
        value = i
        render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, am=value)

    print(np.mean(np.array(timings)))