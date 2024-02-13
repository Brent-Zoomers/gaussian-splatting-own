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

# Current issues (according to me)
"""
Bounding box to small -> undesired results
Threshold for when to add crop is bad/terrible


"""

from segment.sam import get_sam_masks
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, masked_1l_loss, masked_ssim, element_wise_psnr
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, PILtoTorch
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, show_pytorch_image, pytorch2opencv
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import wandb

wandb.init(
    # set the wandb project where this run will be logged
    project="gaussian_splatting",
    
    # track hyperparameters and run metadata
    config={
    "crop_iterations": [2500, 5000],
    "dataset": "truck_big",
    "images": "images_4",
    "epochs": 30_000,
    "num_clusters": 3,
    "ignore_below_surface_percentage": 0,
    "comment": "Changed to 5k10k after seeing 15k stops adding",
    }
)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    # Debug variables
    frames_added = []

    print(opt.densify_viewpoint_iterations)
    densify_viewpoints = [int(''.join(opt.densify_viewpoint_iterations))]
    # print(densify_viewpoints)
    # exit()
    crops_added = 0
    skipped = 0
    # densify_viewpoints = []
    print(densify_viewpoints)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        # Maybe change to N random camera's as crops will each take up one iteration
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            print()
            print(len(viewpoint_stack))
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        # if viewpoint_cam.image_name not in seen:
        #     seen[viewpoint_cam.image_name] = True
            # print(viewpoint_cam.image_name)
        
        # count = 0
        # for x in viewpoint_stack:
        #     gt = x.original_image
        #     cv_gt = pytorch2opencv(gt*viewpoint_stack[0].mask)
        #     count += 1
        #     # if torch.min(viewpoint_stack[0].mask) == 0:
        #     cv2.imshow(f'{count}', cv_gt)
        #     cv2.waitKey()
                



        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        
        # Calculate masked loss

        if torch.min(viewpoint_cam.mask) == 1.0:
            # print("Normal Used")
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            wandb.log({"num_gaussians": gaussians.get_scaling.shape[0],"loss": loss})
            # print(loss.isnan())
            loss.backward()
        else:
            print("Crop Used")
            # calculate masked loss
            mask = viewpoint_cam.mask.cuda()
            gt_image = viewpoint_cam.original_image.cuda()
            binary_mask = torch.where(mask != 0, torch.tensor(1.0).to("cuda"), torch.tensor(0.0).to("cuda"))
            # Calculate the percentage of nonzero items
            percentage = torch.mean(binary_mask)

            cv2.imshow("gt", pytorch2opencv(gt_image))
            cv2.imshow("render", pytorch2opencv(image))
            cv2.imshow("mask", pytorch2opencv(mask))
            cv2.waitKey()

            # show_pytorch_image(mask, "mask")
            # show_pytorch_image(gt_image, "gt_image")
            Ll1 = masked_1l_loss(image, gt_image,mask)
            loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - masked_ssim(image, gt_image,mask))) * (percentage + 0.00001)
            wandb.log({"num_gaussians": gaussians.get_scaling.shape[0],"loss": loss})
            loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # Look for areas which get reconstructed poorly.
                    
            # Render to each viewpoint and look for PSNR below threshold
            if (iteration in densify_viewpoints):
                # viewpoint_stack_ = scene.getTrainCameras().copy()

                viewpoint_stack_ = scene.getInBetweenCameras().copy()
                count = -1

                psnr_list = []
                for camera_info in viewpoint_stack_:
                    camera_info.load_image()
                    camera = cameraList_from_camInfos([camera_info], 1.0, dataset)[0]
                    gt_image = camera.original_image.cuda()
                    camera_info.unload_image()
                    # break
                    render_pkg = render(camera, gaussians, pipe, bg)
                    image, _,_,_ = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

                    # Calculate element-wise PSNR and create binary mask
                    psnr = element_wise_psnr(image, gt_image)

                    grayscale_image = torch.mean(psnr, dim=0, keepdim=True)
                    psnr_list.append(torch.mean(grayscale_image).cpu().numpy())

                mean_psnr = np.mean(psnr_list)
                stdev_psnr = np.std(psnr_list)

                # SAM Test
                for camera_info in viewpoint_stack_:
                    count += 1
                    # if count % 25 != 0:
                    #     continue
                    # print(count)
                    
                    camera_info.load_image()
                    camera = cameraList_from_camInfos([camera_info], 1.0, dataset)[0]
                    gt_image = camera.original_image.cuda()
                    camera_info.unload_image()

                    render_pkg = render(camera, gaussians, pipe, bg)
                    image, _,_,_ = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

                
                    cv_gt_image = pytorch2opencv(gt_image)

                    psnr = element_wise_psnr(image, gt_image)
                    grayscale_image = torch.mean(psnr, dim=0, keepdim=True)

                    if torch.mean(grayscale_image) > mean_psnr - int(opt.stdev_threshold_skipped)*stdev_psnr:
                        continue


                    # Store if non exists and read in if exists
                    masks=[]
                    name = camera.image_name.split("\\")[-1]
                    if os.path.isfile(f'segmentations/{name}.npy'):
                        masks = np.load(f'segmentations/{name}.npy', allow_pickle=True)
                    else:  
                        masks = get_sam_masks(np.clip(cv_gt_image * 255.0, 0, 255).astype(np.uint8))
                        np.savez_compressed(f'segmentations/{name}.npy', masks)

                    psnr = element_wise_psnr(image, gt_image)
                    grayscale_image = torch.mean(psnr, dim=0, keepdim=True)
                    cv_psnr = pytorch2opencv(grayscale_image)

                    final_mask = np.zeros_like(cv_psnr)

                    for mask in masks:
                        uint8_mask = mask['segmentation'].astype(np.uint8)
                        masked_gt = cv_psnr * np.resize(uint8_mask, (uint8_mask.shape[0],uint8_mask.shape[1],1))

                        # Calculate masked PSNR
                        masked_mean_psnr = np.sum(masked_gt) / np.count_nonzero(uint8_mask)
                        print(f'{masked_mean_psnr:,.4f}|{mean_psnr - int(opt.stdev_threshold_segment)*stdev_psnr:,.4f}')
                        if masked_mean_psnr < mean_psnr - int(opt.stdev_threshold_segment)*stdev_psnr:
                            final_mask += np.resize(uint8_mask, (uint8_mask.shape[0],uint8_mask.shape[1],1))
                            

                    final_mask = final_mask > 0
                    show_results = True

                    camera_info.load_image()
                    new_cam = cameraList_from_camInfos([camera_info], 1.0, dataset)
                    camera_info.unload_image()
      
                    new_mask = torch.from_numpy(final_mask.astype(np.uint8)).cuda().permute(2,0,1)
                    new_cam[0].update_mask(new_mask)
                    frames_added.append(new_cam[0].image_name)
                    # # # #
                    scene.addTrainCam(new_cam)
                    scene.deleteInBetweenCam(camera_info)
                   

                    # Add final mask and image to train 

                    # if cv2.waitKey(33) == ord('a'):
                    #     show_results = not show_results
                    #     while cv2.waitKey(33) == ord('a'):
                    #         pass


                    if cv2.waitKey(33) == ord('a'):
                        cv2.imshow("mask", final_mask.astype(np.uint8) * 255)  
                        cv2.imshow("gt", cv_gt_image)  

                        cv2.waitKey()
                        cv2.destroyAllWindows()

                    print("lolol")
    


                        # cv2.imshow("mask", uint8_mask)
                        # cv2.imshow("masked_gt", masked_gt)
                        # cv2.waitKey()
                        # cv2.destroyAllWindows()






                #






                # for camera_info in viewpoint_stack_:
                #     count += 1
                #     if count % 10 != 0:
                #         continue
                    
                #     camera_info.load_image()
                #     camera = cameraList_from_camInfos([camera_info], 1.0, dataset)[0]
                #     gt_image = camera.original_image.cuda()
                #     camera_info.unload_image()
                #     # break
                #     render_pkg = render(camera, gaussians, pipe, bg)
                #     image, _,_,_ = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

                #     # Calculate element-wise PSNR and create binary mask
                #     psnr = element_wise_psnr(image, gt_image)

                #     # Convert to single channel image -> (magnitude?)
                #     grayscale_image = torch.mean(psnr, dim=0, keepdim=True)

                #     print(f'{torch.mean(torch.mean(grayscale_image))} | {mean_psnr - 2*stdev_psnr}')
                #     if torch.mean(torch.mean(grayscale_image)) > mean_psnr - 2*stdev_psnr:
                #         print("Skipped")
                #         continue

                #     # Use mean and stdev to calculate badly reconstructed parts of image
                #     m = torch.mean(grayscale_image)
                #     stdev = torch.std(grayscale_image)
                #     mask = (grayscale_image >= m-stdev).float()

                #     # show_pytorch_image([mask, gt_image], ["mask", "gt_image"])


                #     # Convert to OpenCV for morphological operations + further processing  
                #     numpy_image = pytorch2opencv(mask)
                #     ground_truth = pytorch2opencv(gt_image)

                #     kernel_size = 3
                #     kernel = np.ones((kernel_size, kernel_size), np.uint8)
                #     large_kernel = np.ones((kernel_size+2, kernel_size+2), np.uint8)

                #     # closing = cv2.morphologyEx(numpy_image, cv2.MORPH_CLOSE, kernel)
                #     # erosion = cv2.erode(closing, kernel, iterations=5)

                #     # kernel op 2x2
                #     erosion = cv2.erode(numpy_image, kernel, iterations=1)
                #     # dilate = cv2.dilate(erosion, kernel, iterations=1)
                #     dilate_3 = cv2.dilate(erosion, large_kernel, iterations=3)
                #     d_inv = 1 - dilate_3

                #     # cv2.imshow("???", d_inv)
                #     # cv2.waitKey()


                #     # Detect components (N Largest?, Above X Area?)
                #     rv, lab, stats, centroids = cv2.connectedComponentsWithStats(d_inv.astype(np.uint8))
                #     sizes = stats[:,4 ]
                #     flat_indices = np.argpartition(sizes.flatten(), -11)[-11:]

                #     # Convert flat indices to row, column indices
                #     indices = np.unravel_index(flat_indices, sizes.shape)
                #     non_bg_indices = indices[0][indices[0] != 0]
                #     centroids = centroids[non_bg_indices]

                #     if len(centroids) == 0:
                #         skipped += 1
                #         print(f'Skipped Good reconstruction: {skipped}')
                #         continue

                #     filtered_mask = np.zeros_like(numpy_image, dtype=np.uint8)

                #     for label in non_bg_indices:
                #         # Get the indices of the label in the array
                #         label_indices = np.where(lab == label)
                #         # Set the corresponding elements in the boolean matrix to True
                #         filtered_mask[label_indices] = 255

                #     # if np.mean(mask) == 0:
                #     #     skipped += 1
                #     #     print(f'Skipped empty mask: {skipped}')
                #     #     continue

                #     # Select which images will be picked and transform mask to it?
 
                #     scene.densifyCameras(ground_truth=ground_truth, psnr=grayscale_image,  points=centroids, cam_info=camera_info)
                #     crops_added += 1
                #     print(f'Crops added :{crops_added*5}')



                    # cv2.imshow("mask", numpy_image)
                    # cv2.imshow("gt", ground_truth)
                    # cv2.imshow("morph", erosion)
                    # cv2.waitKey() 

                    # Use K-Means clustering to cluster the coordinates into N clusters
                    # MAX 7 for coloring
                    # zero_coords = np.column_stack(np.nonzero(1-erosion))
                    # if zero_coords.shape[0] == 0:
                    #     skipped += 1
                    #     print(f'Skipped Good reconstruction: {skipped}')
                    #     continue
                    # num_clusters = 6
                    # kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                    # kmeans.fit(zero_coords)

                    # # Get the cluster labels and cluster centers
                    # labels = kmeans.labels_
                    # centers = kmeans.cluster_centers_

                    # # Convert grayscale mask to a 3-channel image
                    # color_mask = ground_truth
                    # mask = np.zeros_like(numpy_image)

                    # # Draw bounding boxes around each cluster on the color image with different colors
                    # for i in range(num_clusters):
                    #     cluster_points = zero_coords[labels == i]
                    #     color = tuple(np.random.randint(0, 256, 3).tolist())  # Random color for each cluster
                    #     for point in cluster_points:
                    #         cv2.circle(color_mask, (point[1], point[0]), 1, color, -1)
                    #     if len(cluster_points) > 0:
                    #         # Calculate bounding box for the cluster
                    #         y_min, x_min = np.min(cluster_points, axis=0)
                    #         y_max, x_max = np.max(cluster_points, axis=0)
                    #         x, y, w, h = int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)

                    #         # if w*h > (erosion.shape[0] * erosion.shape[1]) / 40:
                    #         # Draw bounding box on the color image
                    #         cv2.rectangle(color_mask, (x, y), (x + w, y + h), color, 2)
                    #         cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
                            
                    # cv2.imshow("mask", mask)
                    # cv2.imshow("c_mask", color_mask)
                    # cv2.waitKey()
                    # apply homography to mask where???
                                
                    # if np.mean(mask) == 0:
                    #     skipped += 1
                    #     print(f'Skipped empty mask: {skipped}')
                    #     continue

                    # # Select which images will be picked and transform mask to it?
 
                    # scene.densifyCameras(ground_truth=ground_truth, mask=erosion, camera_position=camera.T, points=zero_coords)
                    # crops_added += 1
                    # print(f'Crops added :{crops_added*5}')
                    # for now if mask is completely black, take complete image
                    # cams = scene.getTrainCameras().copy()
                    # for i in range(2):
                    #     cams[-i].update_mask(mask)
                    # Store mask with image so psnr gets calculated only on masked parts 

                    # saved_mask = (cams[-1].mask.detach().cpu().numpy())     
                    # saved_mask = saved_mask.transpose(1,2,0)
                    # Display the results
                    # cv2.imshow('Original Image', numpy_image)
                    # cv2.imshow('Opened Image', opened_image)
                    # cv2.imshow('Result Image', color_mask)
                    # cv2.imshow('Result Image', mask)
                    # cv2.imshow('Saved_mask', saved_mask)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    
            # Create crop from in-between views and add camera to training list


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
    wandb.finish()
