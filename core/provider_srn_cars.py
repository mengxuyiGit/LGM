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
import einops

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

# exactly the same as self.load_ply() in the the gs.py 
def load_ply(path, compatible=True):

    from plyfile import PlyData, PlyElement

    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    # print("Number of points at loading : ", xyz.shape[0])

    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    shs = np.zeros((xyz.shape[0], 3))
    shs[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    shs[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
    shs[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
    gaussians = np.concatenate([xyz, opacities, scales, rots, shs], axis=1)
    gaussians = torch.from_numpy(gaussians).float() # cpu

    if compatible:
        gaussians[..., 3:4] = torch.sigmoid(gaussians[..., 3:4])
        gaussians[..., 4:7] = torch.exp(gaussians[..., 4:7])
        gaussians[..., 11:] = 0.28209479177387814 * gaussians[..., 11:] + 0.5

    return gaussians

from ipdb import set_trace as st

class SrnCarDataset(Dataset):

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
    
        # load src instrinsic file
        assert len(self.items) == 1
        intrinsic_file = os.path.join(name, 'intrinsics.txt')
        fx, fy, cx, cy, height, width = load_intrinsics(intrinsic_file)
   
       
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

        # default camera intrinsics: objaverse
        # print(f"using fovy:{self.opt.fovy}")
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1
        # print("lgm:",self.proj_matrix)


        # self.proj_matrix = getProjectionMatrix(self.opt.znear, self.opt.zfar, self.opt.fovy, self.opt.fovy).transpose(0,1)
        # print("3dgs:",self.proj_matrix)
        # st()


        self.global_cnt = 0
        
        uid = self.items[0]
        ## load splatter images
        splatter_images_multi_views = []
        splatter_uid = os.path.join(self.opt.resume_workspace, os.path.basename(uid), 'eval_pred_gs_299_0')
        # print("resume_workspace: ", self.opt.resume_workspace)
        
        # find 
        fixed_input_views = np.arange(1,7).tolist()
        vids = fixed_input_views[:self.opt.num_input_views] 
            
        try:
            # print(f"uid:{uid}\nsplatter_uid:{splatter_uid}")
            for input_id in vids[:self.opt.num_input_views]:
                sf = os.path.join(splatter_uid, f"splatter_{input_id-1}.ply")
                if self.opt.verbose:
                    print(f"sf:{sf}")
                splatter_im = load_ply(sf)
                splatter_images_multi_views.append(splatter_im)
                # print(splatter_im.shape) # ([16384, 14])
            
            splatter_images_mv = torch.stack(splatter_images_multi_views, dim=0) # # [6, 16384, 14]
            splatter_images_mv = torch.stack(splatter_images_multi_views, dim=0) # # [6, 16384, 14]
            splatter_res = int(math.sqrt(splatter_images_mv.shape[-2]))
            splatter_images_mv = einops.rearrange(splatter_images_mv, 'v (h w) c -> v c h w', h=splatter_res, w=splatter_res)
            # results['splatters_output'] = splatter_images_mv
            # print(results['splatters_output'].shape) # [6, 14, 128, 128])
            self.splatter_images_mv = splatter_images_mv
            # if getattr(self, 'splatter_images_mv', None) is not None:
                # results['splatters_output'] = self.splatter_images_mv
                # print("getattr(self, 'splatter_images_mv', None) is not None")
        except:
            print("getattr(self, 'splatter_images_mv', None) is None")
            pass

            


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):

        uid = self.items[idx]
        # print(uid)
      
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
        file_pattern = os.path.join(uid, "rgb", f'*{extension}')
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
        fixed_input_views = np.arange(1,7).tolist()
        if self.training:
            vids = fixed_input_views[:self.opt.num_input_views] + np.random.permutation(numerical_value+1).tolist()
            # vids = np.arange(1, numerical_value+1).tolist()
        else:
            vids = fixed_input_views[:self.opt.num_input_views] + np.arange(numerical_value+1).tolist() # fixed order
            # vids = np.arange(1, numerical_value+1).tolist()
        
     
      
        
        # finish 
        if getattr(self, 'splatter_images_mv', None) is not None:
            results['splatters_output'] = self.splatter_images_mv
            
        
        
        # print(vids)
        final_vids = []
        
        for vid in vids:
            final_vids.append(vid)
            
            image_path = os.path.join(uid, 'rgb', f'{vid:06d}.png')
            camera_path = os.path.join(uid, 'pose', f'{vid:06d}.txt')
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
            c2w = load_pose(camera_path)

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
            c2w[:3, 3] *= 1.5 / 1.3
            # print(f"scale the c2w by 1.5 / 1.3")
            # c2w[:3, 3] *= 1.3



            # print(f"scale the c2w by radius={self.opt.cam_radius}/ 2")
            # print(f"scale the c2w by cam_radius / 1.5")
            # c2w[:3, 3:] *= 2
            # print("cam[:3,3] / 0.5")
            
          
            image = image.permute(2, 0, 1) # [4, 512, 512]
            mask = image[3:4] # [1, 512, 512]
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
        # print("cam_poses: ", cam_poses.shape)

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
        # print("cam_view: ",cam_view.shape)
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos

        # results['vids'] = torch.tensor(final_vids)
        results['vids'] = final_vids

        return results