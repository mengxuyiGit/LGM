import os
import cv2
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

import kiui
from core.options import Options
from core.utils import get_rays, grid_distortion, orbit_camera_jitter

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

import glob


class ObjaverseDataset(Dataset):

    def _warn(self):
        raise NotImplementedError('this dataset is just an example and cannot be used directly, you should modify it to your own setting! (search keyword TODO)')

    def __init__(self, opt: Options, name=None, training=True):
        
        self.opt = opt
        self.training = training

        # # TODO: remove this barrier
        # self._warn()

        # TODO: load the list of objects for training
        # self.items = ["/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/data-1000/0c77dfdf9430465f9767a58d56e8fca1"]
        self.items = ["/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/0a9b36d36e904aee8b51e978a7c0acfd"]
        if name is not None:
            self.items = [name]
        # with open('TODO: file containing the list', 'r') as f:
        #     for line in f.readlines():
        #         self.items.append(line.strip())

        # naive split
        # if self.training:
        #     self.items = self.items[:-self.opt.batch_size]
        # else:
        #     self.items = self.items[-self.opt.batch_size:]
        
        # default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1

        self.global_cnt = 0
        # print(f"init self.global_cnt = 0, self.training={self.training}")


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):

        uid = self.items[idx]
        results = {}

        # load num_views images
        images = []
        masks = []
        cam_poses = []
        
        vid_cnt = 0

        # TODO: choose views, based on your rendering settings
        # if self.training:
        #     # input views are in (36, 72), other views are randomly selected
        #     vids = np.random.permutation(np.arange(36, 73))[:self.opt.num_input_views].tolist() + np.random.permutation(100).tolist()
        # else:
        #     # fixed views
        #     vids = np.arange(36, 73, 4).tolist() + np.arange(100).tolist()
        
        # vids = np.arange(1, 10)[:self.opt.num_input_views].tolist() + np.random.permutation(50).tolist()
        # vids = np.arange(0, 7)[:self.opt.num_input_views].tolist()
        
        extension='.png'
        file_pattern = os.path.join(uid, f'*{extension}')
        files = sorted(glob.glob(file_pattern))
        
        if files:
            largest_file = files[-1]
            largest_filename = os.path.splitext(os.path.basename(largest_file))[0]
            numerical_value = int(''.join(c for c in largest_filename if c.isdigit()))
            # print(f"largest_filename:{largest_filename} -> value:{numerical_value}")
        
        # vids = np.random.permutation(numerical_value+1)[:self.opt.num_views].tolist()
        # vids = np.arange(0, numerical_value+1)[:self.opt.num_views].tolist()
        # print(f"vids: {vids}")
        # vids = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        # vids = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
        # vids = [self.global_cnt] * self.opt.num_views
        # self.global_cnt += 1
        
        ## fix input views
        fixed_input_views = np.arange(1,7).tolist()
        if self.training:
            vids = fixed_input_views[:self.opt.num_input_views] + np.random.permutation(numerical_value+1).tolist()
        else:
            vids = fixed_input_views[:self.opt.num_input_views] + np.arange(numerical_value+1).tolist() # fixed order
        
        final_vids = []
        
        for vid in vids:
            final_vids.append(vid)
            
            image_path = os.path.join(uid, f'{vid:03d}.png')
            camera_path = os.path.join(uid, f'{vid:03d}.npy')
            # print(f"image path: {image_path}; cam path:{camera_path}")

            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255 # [512, 512, 4] in [0, 1]
            image = torch.from_numpy(image)

            cam = np.load(camera_path, allow_pickle=True).item()
            # print(f"cam npy contents:{cam}")
            
            from kiui.cam import orbit_camera
            c2w = orbit_camera(-cam['elevation'], cam['azimuth'], radius=cam['radius'])
            c2w = torch.from_numpy(c2w)
            # try:
            #     # TODO: load data (modify self.client here)
            #     image = np.frombuffer(self.client.get(image_path), np.uint8)
            #     image = torch.from_numpy(cv2.imdecode(image, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255) # [512, 512, 4] in [0, 1]
            #     c2w = [float(t) for t in self.client.get(camera_path).decode().strip().split(' ')]
            #     c2w = torch.tensor(c2w, dtype=torch.float32).reshape(4, 4)
            # except Exception as e:
            #     # print(f'[WARN] dataset {uid} {vid}: {e}')
            #     continue
            
            # # TODO: you may have a different camera system
            # # blender world + opencv cam --> opengl world & cam
            # c2w[1] *= -1
            # c2w[[1, 2]] = c2w[[2, 1]]
            # c2w[:3, 1:3] *= -1 # invert up and forward direction

            # scale up radius to fully use the [-1, 1]^3 space!
            c2w[:3, 3] *= self.opt.cam_radius / 1.5 # 1.5 is the default scale
          
            image = image.permute(2, 0, 1) # [4, 512, 512]
            mask = image[3:4] # [1, 512, 512]
            image = image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg
            image = image[[2,1,0]].contiguous() # bgr to rgb

            images.append(image)
            masks.append(mask.squeeze(0))
            cam_poses.append(c2w)

            vid_cnt += 1
            if vid_cnt == self.opt.num_views:
                break
            

        if vid_cnt < self.opt.num_views:
            print(f'[WARN] dataset {uid}: not enough valid views, only {vid_cnt} views found!')
            n = self.opt.num_views - vid_cnt
            images = images + [images[-1]] * n
            masks = masks + [masks[-1]] * n
            cam_poses = cam_poses + [cam_poses[-1]] * n
          
        images = torch.stack(images, dim=0) # [V, C, H, W]
        masks = torch.stack(masks, dim=0) # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]

        # normalized camera feats as in paper (transform the first pose to a fixed position)
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]

        images_input = F.interpolate(images[:self.opt.num_input_views].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        cam_poses_input = cam_poses[:self.opt.num_input_views].clone()

        # data augmentation
        if self.training:
            # apply random grid distortion to simulate 3D inconsistency
            if random.random() < self.opt.prob_grid_distortion:
                images_input[1:] = grid_distortion(images_input[1:])
            # apply camera jittering (only to input!)
            if random.random() < self.opt.prob_cam_jitter:
                print("jjjjjjjjjjiiiiiiiiiitttttttttteeeeeeeerrrrrrrrr")
                cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:])

        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # resize render ground-truth images, range still in [0, 1]
        results['images_output'] = F.interpolate(images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        results['masks_output'] = F.interpolate(masks.unsqueeze(1), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]
        if self.opt.verbose:
            print(f"output size:{results['masks_output'].shape}")

        # build rays for input views
        rays_embeddings = []
        for i in range(self.opt.num_input_views):
            rays_o, rays_d = get_rays(cam_poses_input[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

     
        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
        final_input = torch.cat([images_input, rays_embeddings], dim=1) # [V=4, 9, H, W]
        results['input'] = final_input

        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction

        # should use the same world coord as gs renderer to convert depth into world_xyz
        results['c2w_colmap'] = cam_poses[:self.opt.num_input_views].clone() 
        
        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4], w2c
        cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4], w2pix
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        
        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos

        # results['vids'] = torch.tensor(final_vids)
        results['vids'] = final_vids

        return results