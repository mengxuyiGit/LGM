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
import math

from core.eg3d_camera_utils import LookAtPoseSampler #, FOV_to_intrinsics
from ipdb import set_trace as st

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def load_intrinsics(path):
    with open(path, 'r') as file:
        f, cx, cy, _ = map(float, file.readline().split())
        grid_barycenter = list(map(float, file.readline().split()))
        scale = float(file.readline())
        height, width = map(int, file.readline().split())
    fx = fy = f
    # print(f"fovy of shapenet is {np.rad2deg(focal2fov(fx, height))} and {focal2fov(fy, width)}")
    # print(f"fovy of shapenet is {np.rad2deg(focal2fov(fx, height))} and {focal2fov(fy, width)}")
    # exit()
    return fx, fy, cx, cy, height, width


def load_pose(path):
    pose = np.loadtxt(path, dtype=np.float32, delimiter=' ').reshape(4, 4)
    return torch.from_numpy(pose)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def format_scientific(number):
    # Convert number to scientific notation
    number_str = f"{number:.0e}"
    # Split into base and exponent
    base, exponent = number_str.split('e')
    # Remove leading '+' and extra '0' from the exponent if present
    exponent = exponent.replace('+', '').lstrip('0')
    return f"{base}e{exponent}"


class FFHQDataset(Dataset):

    def _warn(self):
        raise NotImplementedError('this dataset is just an example and cannot be used directly, you should modify it to your own setting! (search keyword TODO)')

    def __init__(self, opt: Options, name=None, training=True):
        
        self.opt = opt
        self.training = training

        # # TODO: remove this barrier
        # self._warn()

        # TODO: load the list of objects for training
        # self.items = ["/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/data-1000/0c77dfdf9430465f9767a58d56e8fca1"]
        if name is None:
            raise NotImplementedError("Please provide a valid data path")
        
        self.items = [name]
        
        # fovy_str = f"{opt.angle_y_step:.0e}" 
        # fovy_str = format_scientific(opt.angle_y_step)
        # if name.split("out_lgm_")[-1] != fovy_str:
        if float(name.split("out_lgm_")[-1]) != opt.angle_y_step:
            print(f"Please provide a data path correspond to the opt.angle_y_step: {opt.angle_y_step}")
            exit(0)

        # load src instrinsic file
        assert len(self.items) == 1
        # intrinsic_file = os.path.join(name, 'intrinsics.txt')
        # fx, fy, cx, cy, height, width = load_intrinsics(intrinsic_file)
   
       
        # # TODO: convert K to all the following stuff
        # # FIXME: ignore the cx cy for now
        # # NOTE :THIS proj_matrix is the transpose of real projection matrix
        # self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        # self.proj_matrix[0, 0] = fx / (width / 2)
        # self.proj_matrix[1, 1] = fy / (height / 2)
        # # NOTE: using default znear/zfar of objaverse, should not matter too much
        # self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        # self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        # self.proj_matrix[2, 3] = 1

        # list25 = [0.9084562063217163, -0.057051945477724075, 0.41406819224357605, -1.0428725481033325, 
        #           0.0, -0.9906408786773682, -0.13649439811706543, 0.3437749147415161, 
        # 0.4179801344871521, 0.12399918586015701, -0.8999537825584412, 2.4666244983673096, 
        # 0.0, 0.0, 0.0, 1.0, # extrinsic
        # 4.465174674987793, 0.0, 0.5, 
        # 0.0, 4.465174674987793, 0.5, 
        # 0.0, 0.0, 1.0]

        # default camera intrinsics: objaverse
        # print(f"using fovy:{self.opt.fovy}")
        # self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        # focal_length = float(1 / (math.tan(fov_degrees * 3.14159 / 360) * 1.414))
        # intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)


        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1
        # print("lgm:",self.proj_matrix)
        # st()


        # self.proj_matrix = getProjectionMatrix(self.opt.znear, self.opt.zfar, self.opt.fovy, self.opt.fovy).transpose(0,1)
        # print("3dgs:",self.proj_matrix)
        # st()

        # cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
        cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2]), radius=2.7)
        # intrinsics = FOV_to_intrinsics(fov_deg, device=device)

        # Generate images.
        angle_p = -0.2
        self.angle_list = [(angle_y, angle_p) for angle_y in np.arange(-.4, .4, opt.angle_y_step)]

        self.global_cnt = 0


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):

        uid = self.items[idx]
        # print("uid: ",uid)
      
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
        file_pattern = os.path.join(uid, f'seed0000_**{extension}')
        files = sorted(glob.glob(file_pattern))

        # print(f"uid:{uid}")
        # print(f"pattern:{file_pattern}")
        
        if files:
            largest_file = files[-1]
            largest_filename = os.path.splitext(os.path.basename(largest_file))[0]
            numerical_value = int(''.join(c for c in largest_filename if c.isdigit()))
            # print(f"largest_filename:{largest_filename} -> value:{numerical_value}")
        else:
            print("Wrong pattern: no file for this scene")
        
        # vids = np.random.permutation(numerical_value+1)[:self.opt.num_views].tolist()
        # vids = np.arange(0, numerical_value+1)[:self.opt.num_views].tolist()
        # print(f"vids: {vids}")
        # vids = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        # vids = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
        # vids = [self.global_cnt] * self.opt.num_views
        # self.global_cnt += 1
        
        ## fix input views
        # mid = numerical_value - 3 # // 2
        mid = numerical_value // 2
        # fixed_input_views = np.arange(1,7).tolist()
        fixed_input_views = np.arange(mid,mid+3).tolist() + np.arange(mid - 3,mid -1).tolist()
        print("fixed_input_views: ", fixed_input_views)
        if self.training:
            vids = fixed_input_views[:self.opt.num_input_views] + np.random.permutation(numerical_value+1).tolist()
            # vids = np.arange(1, numerical_value+1).tolist()
        else:
            vids = fixed_input_views[:self.opt.num_input_views] + np.arange(numerical_value+1).tolist()[::3] # fixed order
            # vids = [] + np.arange(1, numerical_value+1).tolist()[::2]
            print(vids)
        
        # print(vids)
        final_vids = []
        
        for vid in vids:
            final_vids.append(vid)
            
            image_path = os.path.join(uid, f'seed0000_{vid:02d}{extension}')
            # camera_path = os.path.join(uid, 'pose', f'{vid:06d}.txt')
            # print(f"image path: {image_path}; cam path:{camera_path}")


            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255 # [512, 512, 4] in [0, 1]
            image = torch.from_numpy(image)

            # # objaverse
            # cam = np.load(camera_path, allow_pickle=True).item()
            # print(f"cam npy contents:{cam}")
            # from kiui.cam import orbit_camera
            # c2w = orbit_camera(-cam['elevation'], cam['azimuth'], radius=cam['radius'])
            # c2w = torch.from_numpy(c2w)
            
            # for src cars: in opencv
            # c2w = load_pose(camera_path)
            angle_y, angle_p = self.angle_list[vid]
            # print(f"vid={vid}, angle_y, angle_p={angle_y, angle_p}")
           
            cam_pivot = torch.tensor([0, 0, 0])
            cam_radius = 2.7
            cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius)
            # print(f"cam2world_pose: {cam2world_pose.shape}")
            c2w = cam2world_pose[0]

            ## for our objaverse we do not need this
            # # TODO: you may have a different camera system
            # # blender world + opencv cam --> opengl world & cam

            # for srn_cars we may still need this
            c2w[1] *= -1 # [x, -y, z, homo]
            # print("no inverse of z")
            c2w[[1, 2]] = c2w[[2, 1]] # [x, z, -y, homo] .switch: 2nd row <-> 3rd row: world_y <-> world_z
            c2w[:3, 1:3] *= -1 # invert up and forward direction (of cam points: [x', y', z' -> x', -y', -z'], opengl -> opencv): NOTE: this is necessary
            # print("no opengl -> opencv")

            # scale up radius to fully use the [-1, 1]^3 space!
            # c2w[:3, 3] *= self.opt.cam_radius / 1.5 # 1.5 is the default scale
            
            # print(f"cam {vid} radius={torch.norm(c2w[:3, 3])}")
            # c2w[:3, 3] *= 1.69 / 2
            # c2w[:3, 3] *= 2
            # c2w[:3, 3] *= 2 / 1.69
            # c2w[:3, 3] *= 1.5 / 1.69
            # c2w[:3, 3] *= 2 / 1.5
            # c2w[:3, 3] *= 1.69 / 2    
            # c2w[:3, 3] *= 1.3 / 1.5
            # c2w[:3, 3] *= 1.5 / 1.3
            c2w[:3, 3] *= 1.5 / cam_radius
            # print(f"scale the c2w by 1.5 / 1.3")
            # c2w[:3, 3] *= 1.3



            # print(f"scale the c2w by radius={self.opt.cam_radius}/ 2")
            # print(f"scale the c2w by cam_radius / 1.5")
            # c2w[:3, 3:] *= 2
            # print("cam[:3,3] / 0.5")
            
          
            # print(f"image.shape={image.shape}")
            image = image.permute(2, 0, 1) # [4, 512, 512]
            mask = torch.ones_like(image[1:2]) # [1, 512, 512]
            # print("srn image shape",image.shape)
            # print("srn mask",mask)
            image = image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg
            image = image[[2,1,0]].contiguous() # bgr to rgb

            images.append(image)
            masks.append(mask.squeeze(0))
            cam_poses.append(c2w)

            vid_cnt += 1
            if vid_cnt == self.opt.num_views:
                break

        # exit()  

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
        # transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1.5], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        # transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]
        # print("Hardcode transform radius = 1.5")


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