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


import glob
import einops
import math
import json

from ipdb import set_trace as st

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def save_all_56_in_1(image_dir):
    # Get a sorted list of all the PNG files in the directory
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png") and f.split('.')[0].isdigit() and 0 <= int(f.split('.')[0]) <= 55])

    # Load all the images into a list
    images = [Image.open(os.path.join(image_dir, file)) for file in image_files]

    # Determine the size for the final big image
    total_width = images[0].width * 8  # 8 images per row
    total_height = images[0].height * (len(images) // 8 + (1 if len(images) % 8 != 0 else 0))  # Calculate the number of rows

    # Create a new blank image with the appropriate size
    big_image = Image.new("RGB", (total_width, total_height))

    # Paste each image into the big image
    for idx, img in enumerate(images):
        x_offset = (idx % 8) * img.width
        y_offset = (idx // 8) * img.height
        big_image.paste(img, (x_offset, y_offset))

    # Save the big image
    big_image_path = os.path.join(f"big_image/{os.path.basename(image_dir)}.png")
    big_image.save(big_image_path)

    print("Big image saved at:", big_image_path)

# exactly the same as self.load_ply() in the the gs.py 
def save_ply(path):
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

import re
def extract_first_number(folder_name):
    match = re.search(r'\d+', folder_name)
    return int(match.group()) if match else None

def return_final_scene(scene_workspace, acceptable_epoch, verbose=False):
  
    for item in os.listdir(scene_workspace):
        if item.endswith("encoder_input"):
            continue
        
        item_epoch = extract_first_number(item)
        # if item_epoch is None or item.startswith('events'):
        if item_epoch is None or not os.path.isdir(os.path.join(scene_workspace, item)):
            continue

        # print(f"extract first number from item {item}: ",extract_first_number(item))
        # print(item)
        # print(item_epoch)
        if item.endswith('_success'):
            if verbose:
                print(f"Already early stopped.")
            return item
            
        elif item_epoch>=acceptable_epoch:# already achieved the max training epochs
            if verbose:
                print(f"Already achieved the acceptable training epochs.")
            # check content
            # return the correct folder
            return item
    return None

    # ---------
from torchvision import transforms
from PIL import Image
from pytorch3d.transforms import quaternion_to_axis_angle, axis_angle_to_quaternion

# process the loaded splatters into 3-channel images
gt_attr_keys = ['pos', 'opacity', 'scale', 'rotation', 'rgbs']
start_indices = [0, 3, 4, 7, 11]
end_indices = [3, 4, 7, 11, 14]
attr_map = {key: (si, ei) for key, si, ei in zip (gt_attr_keys, start_indices, end_indices)}
ordered_attr_list = ["pos", # 0-3
                'opacity', # 3-4
                'scale', # 4-7
                "rotation", # 7-11
                "rgbs", # 11-14
            ] # must be an ordered list according to the channels

sp_min_max_dict = {
    "pos": (-0.7, 0.7), 
    "scale": (-10., -2.),
    "rotation": (-3., 3.)
    }

def load_splatter_mv_ply_as_dict(splatter_dir, device="cpu"):
    
    # splatter_mv = torch.load(os.path.join(splatter_dir, "splatters_mv.pt")).detach().cpu() # [14, 384, 256]
    splatter_mv = torch.load(os.path.join(splatter_dir, "splatters_mv.pt"), map_location='cpu').detach().cpu()
        
    # splatter_mv = torch.load("splatters_mv_02.pt")[0]
    # print("Loading splatters_mv:", splatter_mv.shape) # [1, 14, 384, 256]

    splatter_3Channel_image = {}
            
    for attr_to_encode in ordered_attr_list:
        # print("latents_all_attr_list <-",attr_to_encode)
        si, ei = attr_map[attr_to_encode]
        
        sp_image = splatter_mv[si:ei]
        # print(f"{attr_to_encode}: {sp_image.min(), sp_image.max()}")

        #  map to 0,1
        if attr_to_encode in ["pos"]:
            sp_min, sp_max = sp_min_max_dict[attr_to_encode]
            sp_image = (sp_image - sp_min)/(sp_max - sp_min)
        elif attr_to_encode == "opacity":
            sp_image = sp_image.repeat(3,1,1)
        elif attr_to_encode == "scale":
            sp_image = torch.log(sp_image)
            sp_min, sp_max = sp_min_max_dict[attr_to_encode]
            sp_image = (sp_image - sp_min)/(sp_max - sp_min)
            sp_image = sp_image.clip(0,1)
        elif  attr_to_encode == "rotation":
            # print("processing rotation: ", si, ei)
            assert (ei - si) == 4
            quat = einops.rearrange(sp_image, 'c h w -> h w c')
            axis_angle = quaternion_to_axis_angle(quat)
            sp_image = einops.rearrange(axis_angle, 'h w c -> c h w')
            # print(f"{attr_to_encode}: {sp_image.min(), sp_image.max()}")
            # sp_min, sp_max = -3, 3
            sp_min, sp_max = sp_min_max_dict[attr_to_encode]
            sp_image = (sp_image - sp_min)/(sp_max - sp_min)
        elif attr_to_encode == "rgbs":
            pass
        
        # map to [-1,1]
        sp_image = sp_image * 2 - 1
        sp_image = sp_image.clip(-1,1)
        
        # print(f"{attr_to_encode}: {sp_image.min(), sp_image.max(), sp_image.shape}")
        assert sp_image.shape[0] == 3
        splatter_3Channel_image[attr_to_encode] = sp_image.detach().cpu()
    
    return splatter_3Channel_image

class ObjaverseDataset(Dataset):

    def __init__(self, opt: Options, training=True, prepare_white_bg=False):
        
        self.opt = opt
        if self.opt.model_type == 'LGM':
            self.opt.bg = 1.0
        self.training = training
        self.prepare_white_bg = prepare_white_bg

        self.data_path_rendering = {}
        self.data_path_vae_splatter = {}
 
        excluded_splits = ["40000-49999"] # used for test
        included_splits = [split for split in os.listdir(opt.data_path_rendering) if split not in excluded_splits]

        scene_path_patterns = [os.path.join(opt.data_path_vae_splatter, split, "*", "splatters_mv_inference", "*") for split in included_splits]
       
        all_scene_paths = []
        for pattern in scene_path_patterns:
            all_scene_paths.extend(sorted(glob.glob(pattern)))
        
        # st()
        # scene_path_pattern = os.path.join(opt.data_path_vae_splatter, "*", "*", "splatters_mv_inference", "*")
        # all_scene_paths = sorted(glob.glob(scene_path_pattern)) # 44815 in total. And sorted by the absolute path
        # st()
        
        # remove invalid uids
        if opt.invalid_list is not None:
            print(f"Filter invalid objects by {opt.invalid_list}")
            with open(opt.invalid_list) as f:
                invalid_objects = json.load(f)
            invalid_objects = [os.path.basename(o).replace(".glb", "") for o in invalid_objects]
        else:
            invalid_objects = []
        
            
        for scene_path in all_scene_paths:

            if self.opt.overfit_one_scene and len(self.data_path_vae_splatter) > 3:
                break
   
            if not os.path.isdir(scene_path):
                continue
    
            scene_name = scene_path.split('/')[-1]
            scene_range = scene_path.split('/')[-4]
            # print("scene name:", scene_name)
            if scene_name.split("_")[-1] in invalid_objects:
                rendering_folder = os.path.join(opt.data_path_rendering, scene_range, scene_name.split("_")[-1])
                # print(f"{rendering_folder} is invalid")
                continue 
            
            if scene_name in self.data_path_vae_splatter.keys():
                continue
            
            if not os.path.exists(os.path.join(scene_path, "splatters_mv.pt")):
                continue
            
            self.data_path_vae_splatter[scene_name] = scene_path
            rendering_folder = os.path.join(opt.data_path_rendering, scene_range, scene_name.split("_")[-1])
            self.data_path_rendering[scene_name] = rendering_folder  
               
        assert len(self.data_path_vae_splatter) == len(self.data_path_rendering)

        
        all_items = [k for k in self.data_path_vae_splatter.keys()]

         # naive split
        if self.training:
            self.items = all_items[:-2*10]
        else:
            self.items = all_items[-2*10:]

        if self.opt.overfit_one_scene:
            self.items = all_items[0:1]

        print(f"There are total {len(self.items)} in dataloader")
        

        # default camera intrinsics
        assert self.opt.fovy==60 # now that we use finetuned LGM splatter, the fovy must be 60
        
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        
        scene_name = self.items[idx]
        # if self.opt.overfit_one_scene:
        #     print(f"[WARN]: always fetch the {idx} item. For debug use only")
        
        uid = self.data_path_rendering[scene_name]
        splatter_uid = self.data_path_vae_splatter[scene_name] 
        if self.opt.verbose:
            print(f"uid:{uid}\nsplatter_uid:{splatter_uid}")
        
        results = {}

        # load num_views images
        images = []
        images_white = []
        masks = []
        cam_poses = []
        
        vid_cnt = 0

        # if self.training:
        #     vids = np.arange(1, 7)[:self.opt.num_input_views].tolist() + np.random.permutation(56).tolist()
        # else:
        #     vids = np.arange(1, 7)[:self.opt.num_input_views].tolist() + np.arange(7, 56).tolist()

      
        if self.opt.num_input_views == 4:
            vids = [np.arange(1, 7)[_index] for _index in [0, 2, 4, 5]]
        elif self.opt.num_input_views == 6:
            vids = np.arange(1, 7) # only load the 6 views
        else:
            raise NotImplementedError
        # print(vids)

        cond_path = os.path.join(uid, f'000.png')
        # cond = to_rgb_image(Image.open("/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/data_test/anya_rgba.png"))
        # cond_path = "/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/data_test/anya_rgba.png"
        from PIL import Image
        cond = np.array(Image.open(cond_path).resize((self.opt.input_size, self.opt.input_size)))
        # print(f"cond size:{Image.open(cond_path)}")
        mask = cond[..., 3:4] / 255
        cond = cond[..., :3] * mask + (1 - mask) * int(self.opt.bg * 255)
        results['cond'] = cond.astype(np.uint8)

        # splatter_original_Channel_mvimage_dict = load_splatter_mv_ply_as_dict(splatter_uid)
        # if self.opt.train_unet_single_attr is not None:
        #     for attr in self.opt.train_unet_single_attr:
        #         results[attr] = splatter_original_Channel_mvimage_dict[attr]
        # else:
        #     results.update(splatter_original_Channel_mvimage_dict)
        
        for vid in vids:

            image_path = os.path.join(uid, f'{vid:03d}.png')
            camera_path = os.path.join(uid, f'{vid:03d}.npy')

            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255 # [512, 512, 4] in [0, 1]
            # print("images shape: ", image.shape) # 320x320x4 for LVIS 46K too
            image = torch.from_numpy(image)

            cam = np.load(camera_path, allow_pickle=True).item()
            # print(f"{vid} - elevation: {cam['elevation']}, azimuth: {cam['azimuth']}")
            c2w = orbit_camera(-cam['elevation'], cam['azimuth'], radius=cam['radius'])
            c2w = torch.from_numpy(c2w)
            c2w[:3, 3] *= 1.5 / self.opt.cam_radius  # 1.5 is the default scale
          
            image = image.permute(2, 0, 1) # [4, 512, 512]
            mask = image[3:4] # [1, 512, 512]
            
            if self.prepare_white_bg:
                image_white = image[:3] * mask + (1 - mask) * 1.0
                image_white = image_white[[2,1,0]].contiguous() # bgr to rgb
                images_white.append(image_white)
            
            image = image[:3] * mask + (1 - mask) * self.opt.bg # [3, 512, 512], to white bg
            image = image[[2,1,0]].contiguous() # bgr to rgb
            images.append(image)
            
            masks.append(mask.squeeze(0))
            cam_poses.append(c2w)

            vid_cnt += 1
            # if self.training and (vid_cnt == self.opt.num_views):
            if (vid_cnt == self.opt.num_views):
                break

        # if vid_cnt < self.opt.num_views:
        #     print(f"vid_cnt{vid_cnt} < self.opt.num_views{self.opt.num_views}")
        #     print(f'[WARN] dataset {uid}: not enough valid views, only {vid_cnt} views found!')
        #     n = self.opt.num_views - vid_cnt
        #     images = images + [images[-1]] * n
        #     if self.prepare_white_bg:
        #         images_white = images_white + [images_white[-1]] * n
        #     masks = masks + [masks[-1]] * n
        #     cam_poses = cam_poses + [cam_poses[-1]] * n
          
        images = torch.stack(images, dim=0) # [V, C, H, W]
        if self.prepare_white_bg:
            images_white = torch.stack(images_white, dim=0) # [V, C, H, W]
        masks = torch.stack(masks, dim=0) # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]

        # normalized camera feats as in paper (transform the first pose to a fixed position)
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1.5], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0]) # w2c_1
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4], c2c_1

       
        images_input = F.interpolate(images[:self.opt.num_input_views].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        masks_input = F.interpolate(masks[:self.opt.num_input_views].clone().unsqueeze(1), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        # if self.prepare_white_bg:
        #     images_input_white = F.interpolate(images_white[:self.opt.num_input_views].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        # cam_poses_input = cam_poses[:self.opt.num_input_views].clone()

        # FIXME: we don't need this for zero123plus?
        if self.opt.model_type == 'LGM':
            images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        results['input'] = einops.rearrange(images_input, "(m n) c h w -> c (m h) (n w)", n=2) * 2 - 1 # maps to [-1,1]
        results['masks_input'] = einops.rearrange(masks_input.repeat(1,3,1,1), "(m n) c h w -> c (m h) (n w)", n=2) * 2 - 1 # maps to [-1,1]
        # results['input_white'] = images_input_white

        # resize render ground-truth images, range still in [0, 1]
        render_input_views = self.opt.render_input_views
        
     
        results['images_output'] = F.interpolate(images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        # results['masks_output'] = F.interpolate(masks.unsqueeze(1), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]
        if 'masks_input' in self.opt.train_unet_single_attr and len(self.opt.train_unet_single_attr)==1:
            results['masks_output'] = masks_input.repeat(1,3,1,1)
        elif 'masks_input' in self.opt.train_unet_single_attr and len(self.opt.train_unet_single_attr)==2:
            results['images_output'] = torch.cat([results['images_output'], masks_input.repeat(1,3,1,1)])
            # print("mask input as images out")
       
       
        # lgm_images_input = F.interpolate(images[:self.opt.num_input_views].clone(), size=(256, 256), mode='bilinear', align_corners=False) # [V, C, H, W]
        # lgm_images_input = TF.normalize(lgm_images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # # opengl to colmap camera for gaussian renderer
        # cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction

        # # should use the same world coord as gs renderer to convert depth into world_xyz
        # results['c2w_colmap'] = cam_poses[:self.opt.num_input_views].clone() 

        # # cameras needed by gaussian rasterizer
        # cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        # cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
        
        # cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        # if render_input_views:
        #     results['cam_view'] = cam_view
        #     results['cam_view_proj'] = cam_view_proj
        #     results['cam_pos'] = cam_pos

        # else:
        #     results['cam_view'] = cam_view[self.opt.num_input_views:]
        #     results['cam_view_proj'] = cam_view_proj[self.opt.num_input_views:]
        #     results['cam_pos'] = cam_pos[self.opt.num_input_views:]

        results['scene_name'] = scene_name
        # print(f"returning scene_name:{scene_name}")
        
        # for attr_to_encode in ordered_attr_list:
        #     sp_image = results[attr_to_encode]
        #     print(f"[end of dataloader]{attr_to_encode}: {sp_image.min(), sp_image.max()}")
            
        return results