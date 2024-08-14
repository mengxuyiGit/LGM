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
from kiui.cam import orbit_camera
from core.options import Options
from core.utils import get_rays, grid_distortion, orbit_camera_jitter

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

import glob
from ipdb import set_trace as st

class ObjaverseDataset(Dataset):

    def _warn(self):
        raise NotImplementedError('this dataset is just an example and cannot be used directly, you should modify it to your own setting! (search keyword TODO)')

    def __init__(self, opt: Options, name=None, training=True):
        
        self.opt = opt
        self.training = training
      
        self.data_path_rendering = {}

        excluded_splits = ["40000-49999"] # used for test
        included_splits = [split for split in os.listdir(opt.data_path_rendering) if split not in excluded_splits]
        scene_path_patterns = [os.path.join(opt.data_path_rendering, split, "*") for split in included_splits]
        all_scene_paths = []
        for pattern in scene_path_patterns:
            all_scene_paths.extend(sorted(glob.glob(pattern)))
            # break # for fast dev
        
        for i, scene_path in enumerate(all_scene_paths):
            if i > 1:
                break
            
            scene_name = scene_path.split('/')[-1]
            if not os.path.isdir(scene_path):
                continue
            self.data_path_rendering[scene_name] = scene_path

        self.items = [k for k in self.data_path_rendering.keys()]
        
        if self.opt.overfit_one_scene:
            _oi = 1
            self.items = self.items[_oi:_oi+1]*1001*self.opt.batch_size

         # naive split
        if self.training:
            self.items = self.items[:-self.opt.batch_size]
        else:
            self.items = self.items[-self.opt.batch_size:]
        print(f"Total {len(self.items)} in {'train' if self.training else 'test'} dataloader")

        
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

        uid = self.data_path_rendering[self.items[idx]]
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

        # print("uid = ", uid)
        
        if files:
            largest_file = files[-1]
            largest_filename = os.path.splitext(os.path.basename(largest_file))[0]
            numerical_value = int(''.join(c for c in largest_filename if c.isdigit()))
            # print(f"largest_filename:{largest_filename} -> value:{numerical_value}")
            
        ## fix input views
        fixed_input_views = np.arange(1,7).tolist()
        assert self.opt.num_input_views == 6
        if self.training:
            vids = fixed_input_views[:self.opt.num_input_views] + np.random.permutation(numerical_value+1).tolist()
        else:
            vids = fixed_input_views[:self.opt.num_input_views] + np.arange(numerical_value+1).tolist() # fixed order
            print(vids)
        
        final_vids = []
        
        
        for vid in vids:
            final_vids.append(vid)
            
            image_path = os.path.join(uid, f'{vid:03d}.png')
            camera_path = os.path.join(uid, f'{vid:03d}.npy')
            # print(f"image path: {image_path}; cam path:{camera_path}")

            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255 # [512, 512, 4] in [0, 1]
            image = torch.from_numpy(image)

            cam = np.load(camera_path, allow_pickle=True).item()
            # print(f"cam npy contents:{cam['azimuth']}")
            
            c2w = orbit_camera(-cam['elevation'], cam['azimuth'], radius=cam['radius'])
            c2w = torch.from_numpy(c2w)

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
        results['scene_name'] = uid.split('/')[-1]

        return results