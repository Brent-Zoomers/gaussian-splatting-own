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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import torch

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
    
    def findHomography(self, image1, image2):
        import cv2
        import numpy as np
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

        img_matches = cv2.drawMatches((image1*255).astype(np.uint8), keypoints1, (image2*255).astype(np.uint8), keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display or save the image with matches
        # cv2.imshow('Matches', img_matches)
        # cv2.waitKey(0)

        return H
    
    def densifyCameras(self, scale=1.0, ground_truth=None, mask=None, camera_position=(0,0,0)):
        # Pop camera(s) from in-between standpoints that are close to the badly reconstructed view

        from scipy.spatial import KDTree
        import cv2
        import numpy as np
        positions = [tuple(x.T) for x in self.scene_info.in_between_cameras]
        cameras = [x for x in self.scene_info.in_between_cameras]
        if len(positions) == 0:
            return
        # positions = [tuple(x) for x in positions]
        self.kdtree = KDTree(positions)
        amount_to_return = 2
        idxs = []
        if len(positions) == 0:
            return
        if len(positions) < amount_to_return:
            idxs = self.kdtree.indices
        else:
            _, idxs = self.kdtree.query(camera_position,k=amount_to_return)

        print(idxs)
        
        for idx in idxs:
            cameras[idx].load_image()
          
    
            # Calculate homography, apply to mask and store it somewhere

            # Convert from pytorch to opencv
            numpy_image = np.array(cameras[idx].image)
            # numpy_image = numpy_image.transpose(1,2,0)
            
            H = self.findHomography(ground_truth, numpy_image)

            # print(camera_position, positions[idx])
            warped_mask = cv2.warpPerspective(mask, H, (numpy_image.shape[1], numpy_image.shape[0]), borderValue=0)

            # cv2.imshow("gt", ground_truth)
            # cv2.imshow("new_image", numpy_image)
            # cv2.imshow("wp", warped_mask)
            # cv2.imshow("mask", mask)
            # cv2.waitKey()
            # convert to tensor

            # Apply homography to mask


            # Perform operation based on bounding boxes to add crop in stead of 
            new_cam = cameraList_from_camInfos([cameras[idx]], scale, self.args)
            #
            warped_mask_torch = torch.from_numpy(warped_mask/255.0).cuda()
            binary_warped_mask_torch = torch.where(warped_mask_torch != 0, torch.tensor(1).cuda(), warped_mask_torch)
            new_cam[0].update_mask(binary_warped_mask_torch)
            #
            self.train_cameras[scale].extend(new_cam)
            self.scene_info.in_between_cameras.remove(cameras[idx])

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

    