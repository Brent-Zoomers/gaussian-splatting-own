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
from segment.sam import get_sam_masks

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import torch
from scipy.spatial import KDTree
import cv2
import numpy as np

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """

        self.args = args
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            self.scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            self.scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(self.scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if self.scene_info.test_cameras:
                camlist.extend(self.scene_info.test_cameras)
            if self.scene_info.train_cameras:
                camlist.extend(self.scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(self.scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(self.scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = self.scene_info.nerf_normalization["radius"]

        self.in_between_cameras = {}
        self.train_cameras = {}
        self.test_cameras = {}

        # Create data structure to efficiently query closest camera
       
        # Decide which images will be used n load in train and test cameras
        for cam in self.scene_info.train_cameras:
            cam.load_image()
        for cam in self.scene_info.test_cameras:
            cam.load_image()

        for resolution_scale in resolution_scales: # Loop not used for now
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.test_cameras, resolution_scale, args)
            print("Loading Im Between Cameras")
            # self.in_between_cameras[resolution_scale] = cameraList_from_camInfos(self.scene_info.in_between_cameras, resolution_scale, args)
            # print()

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(self.scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getInBetweenCameras(self, scale=1.0):
        return self.scene_info.in_between_cameras
    
    def addTrainCam(self, cam, scale=1.0):
        self.train_cameras[scale].extend(cam)

    def deleteInBetweenCam(self, camera_info):
        self.scene_info.in_between_cameras.remove(camera_info)

    
    def findHomography(self, image1, image2):
       
        # image1 = image1_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        # image2 = image2_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        # cv2.imshow("1", image1)
        # cv2.imshow("2", image2)
        # cv2.waitKey()

        # Load the mask
        # mask = cv2.imread('mask.jpg', cv2.IMREAD_GRAYSCALE)

        # Detect ORB keypoints and descriptors
        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute((image1*255).astype(np.uint8), None)
        keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

        # Use a Brute Force matcher with Hamming distance
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)

        # Sort the matches based on their distances
        matches = sorted(matches, key=lambda x: x.distance)

        # Get the matching key points
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Calculate the homography matrix using RANSAC
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # img_matches = cv2.drawMatches((image1*255).astype(np.uint8), keypoints1, (image2*255).astype(np.uint8), keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display or save the image with matches
        # cv2.imshow('Matches', img_matches)
        # cv2.waitKey(0)

        return H
    
    def densifyCameras(self, scale=1.0, ground_truth=None, cam_info=None, psnr=None, points=None):


        points_ = np.array([[x,y] for (y,x) in points])
        numpy_image = np.array(ground_truth)
        masks = get_sam_masks(np.clip(numpy_image * 255.0, 0, 255).astype(np.uint8))

        masks_added_idx = []
        for centroid in points_:
            j = 0
            score = []
            mask_idx = []
            area = ground_truth.shape[0]*ground_truth.shape[1]
            curr_best_idx = 0
            for mask_ in masks:
                # Check if mask contains centroid and find smallest one
                
                segmentation = mask_['segmentation']
            
                img = np.resize(segmentation.astype(np.uint8) * 255, (segmentation.shape[0],segmentation.shape[1], 1))
                for point in points_:
                    point = int(point[1]), int(point[0])
                    cv2.circle(img, tuple(point), 5, 127, -1)
                # cv2.imshow("curr_mask", img)

                # score = mask_['predicted_iou']
                if segmentation[int(centroid[0])][int(centroid[1])]:
                    m_area = mask_['area']
                    if m_area < area:
                        curr_best_idx = j
                        area = m_area
                        # masks_added_idx.append(j)
                        # score.append(mask_['predicted_iou'])     
                j+=1
            masks_added_idx.append(curr_best_idx)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            # mask_ = masks[kdTree.query(centroid)[1]]
            
        # TODO Extract useful masks from sam and get bounding box -> store bounding box as crop + extra info    

        result = np.full(segmentation.shape, False, dtype=bool)  

        for mask__idx in set(masks_added_idx):
            mask_ = masks[mask__idx]
            result = np.bitwise_or(result, mask_['segmentation'])

        # calculate psnr on masked parts
            
        masked_psnr = result.astype(np.float32) * np.resize(psnr.cpu().numpy(), result.shape)

        mean_masked_psnr = np.sum(masked_psnr) / np.count_nonzero(result.astype(np.float32))
        print(f'Mean masked psnr is {mean_masked_psnr}')
        # if mean_masked_psnr > 0.30:
        #     return
 

        # img = result.astype(np.uint8)
        # img *= 255
        # cv2.imshow("lol", img)
        # cv2.imshow("gt", ground_truth)
         
        
        # Show added masks
        k = 0
        for mask__idx in set(masks_added_idx):
            mask_ = masks[mask__idx]
            img = np.resize(mask_['segmentation'].astype(np.uint8) * 255, (mask_['segmentation'].shape[0],mask_['segmentation'].shape[1], 1))
            for point in points_:
                point = int(point[1]), int(point[0])
                cv2.circle(img, tuple(point), 3, 127, -1)  # Grey circles
                
            
            # cv2.imshow(str(k) , img)
            k+=1
        # cv2.waitKey()
        # cv2.destroyAllWindows()  
            # cv2.waitKey()
            # cv2.destroyAllWindows()   
            
    
            # Use centroids for SAM :D

            

            # Loop through masks and find potential ones
            # cv2.imshow("gt", ground_truth)
            # cv2.imshow("new_image", numpy_image)
            # cv2.imshow("wp", warped_mask)
            # cv2.imshow("mask", mask)
            # cv2.imshow("sam", np.uint8(masks[0]['segmentation'])*255)
            # cv2.waitKey()
            # cv2.destroyAllWindows()

            # Perform operation based on bounding boxes to add crop in stead of 
        cam_info.load_image()
        new_cam = cameraList_from_camInfos([cam_info], scale, self.args)
        cam_info.unload_image()
        # #
        # warped_mask_torch = torch.from_numpy(warped_mask/255.0).cuda()
        # binary_warped_mask_torch = torch.where(warped_mask_torch != 0, torch.tensor(1).cuda(), warped_mask_torch)

        new_cam[0].update_mask(torch.from_numpy(result/255.0).cuda())
        # #
        self.train_cameras[scale].extend(new_cam)
            # self.scene_info.in_between_cameras.remove(cameras[idx])
        self.scene_info.in_between_cameras.remove(cam_info)
            # cameras[idx].unload_image()

        # cv2.imshow("mask", mask)
        # cv2.imshow("warped mask", warped_mask)
        # cv2.imshow("numpy", numpy_image)
        # cv2.imshow("gt", ground_truth)
        # cv2.waitKey()

        # # For now -> random
        # for _ in range(10):
        #     new_cam = self.scene_info.in_between_cameras.pop(random.randint(0, len(self.scene_info.in_between_cameras) - 1))
        #     new_cam.load_image()
        #     self.train_cameras[scale].extend(cameraList_from_camInfos([new_cam], scale, self.args))

    