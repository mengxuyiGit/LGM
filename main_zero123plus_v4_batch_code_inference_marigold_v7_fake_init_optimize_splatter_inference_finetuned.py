import tyro
import time
import random

import torch
from core.options import AllConfigs
from core.models_zero123plus import Zero123PlusGaussian, gt_attr_keys, start_indices, end_indices, fuse_splatters
from core.models_zero123plus_code import Zero123PlusGaussianCode
from core.models_zero123plus_code_unet_lora import Zero123PlusGaussianCodeUnetLora

from core.models_fix_pretrained import LGM

from accelerate import Accelerator, DistributedDataParallelKwargs
from safetensors.torch import load_file
from core.dataset_v4_batch import ObjaverseDataset as Dataset

import kiui
from datetime import datetime
import torch.utils.tensorboard as tensorboard
import shutil, os

from ipdb import set_trace as st
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
import torch.nn.functional as F

import warnings
from accelerate.utils import broadcast
import re

import numpy
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
# from zero123plus.img_to_mv_v3_my_decoder import to_rgb_image, unscale_image, unscale_latents

import einops
import rembg
import requests
import glob

from pytorch3d.transforms import quaternion_to_axis_angle, axis_angle_to_quaternion

from itertools import islice


def to_rgb_image(maybe_rgba: Image.Image):
    if maybe_rgba.mode == 'RGB':
        return maybe_rgba
    elif maybe_rgba.mode == 'RGBA':
        rgba = maybe_rgba
        img = numpy.random.randint(255, 256, size=[rgba.size[1], rgba.size[0], 3], dtype=numpy.uint8)
        img = Image.fromarray(img, 'RGB')
        img.paste(rgba, mask=rgba.getchannel('A'))
        return img
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)

def unscale_latents(latents):
    latents = latents / 0.75 + 0.22
    return latents

def unscale_image(image):
    image = image / 0.5 * 0.8
    return image

def scale_image(image):
    image = image * 0.5 / 0.8
    return image

def scale_latents(latents):
    latents = (latents - 0.22) * 0.75
    return latents

def normalize_to_target(source_tensor, target_tensor):
    # Calculate mean and standard deviation of source tensor
    source_mean = torch.mean(source_tensor)
    source_std = torch.std(source_tensor)

    # Calculate mean and standard deviation of target tensor
    target_mean = torch.mean(target_tensor)
    target_std = torch.std(target_tensor)

    # Normalize source tensor to target distribution
    normalized_tensor = (source_tensor - source_mean) / source_std * target_std + target_mean

    return normalized_tensor

def fuse_splatters(splatters):
    # fuse splatters
    B, V, C, H, W = splatters.shape

    x = splatters.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)
    return x

### Functions for vae_encode splatter images
## depth <-> pose
import math
def init_ray_dirs(opt):
    res = opt.output_size

    x = torch.linspace(-res // 2 + 0.5, 
                        res // 2 - 0.5, 
                        res)
    y = torch.linspace( res // 2 - 0.5, 
                        -res // 2 + 0.5, 
                        res)
    ## NOTE: we assume using colmap to match 3dgs, thus need to invert y
    inverted_x = False
    inverted_y = True
    if inverted_x:
        x = -x
    if inverted_y:
        y = -y

    grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
    ones = torch.ones_like(grid_x, dtype=grid_x.dtype) # NOTE: should be ok
    print("Use - ones for homo")
    # st()
    ray_dirs = torch.stack([grid_x, grid_y, ones]).unsqueeze(0)

    # for cars and chairs the focal length is fixed across dataset
    # so we can preprocess it
    # for co3d this is done on the fly
    # if self.cfg.data.category == "cars" or self.cfg.data.category == "chairs" \
    #     or self.cfg.data.category == "objaverse":
        # ray_dirs[:, :2, ...] /= fov2focal(self.cfg.data.fov * np.pi / 180, 
        #                                   self.cfg.data.training_resolution)
    
    focal = res * 0.5 / math.tan(0.5 * np.deg2rad(opt.fovy)) # or change to torch.deg2tan
    ray_dirs[:, :2, ...] /= focal
    # self.register_buffer('ray_dirs', ray_dirs) # [1 3 128 128]
    return ray_dirs


    # ray_dists_sq = torch.sum(ray_dirs**2, dim=1, keepdim=True)
    # ray_dists = ray_dists_sq.sqrt() # [1 1 128 128]
    # self.register_buffer('ray_dists', ray_dists) # [1 3 128 128]


def get_world_xyz_from_depth_offset(depth_activated, xyz_offset, data, opt):
    
    B, V, C, H, W = depth_activated.shape
    print("depth_activated.shape: ", depth_activated.shape)

    depth = einops.rearrange(depth_activated, 'b v c h w -> (b v) (h w) c', b=B, v=V, h=H, w=W)
    xyz_offset = einops.rearrange(xyz_offset, 'b v c h w -> (b v) (h w) c', b=B, v=V, h=H, w=W)

    global_ray_dirs = init_ray_dirs(opt)
    batch_ray_dirs = einops.rearrange(global_ray_dirs, 'b c h w -> b (h w) c').repeat(depth.shape[0], 1, 1)

    # cam view depth + cam view offset -> cam xyz 

    cam_xyz2 = batch_ray_dirs.to(depth.device) * depth + xyz_offset
    
    # c2w 
    cam_xyz_homo2 = torch.cat([cam_xyz2, torch.ones_like(cam_xyz2[...,-1:])], dim=-1)

    c2w = einops.rearrange(data['c2w_colmap'].transpose(-2, -1), 'b v row col -> (b v) row col')
    world_xyz_homo2 = torch.bmm(cam_xyz_homo2, c2w)
    world_xyz2 = world_xyz_homo2[...,:3] / world_xyz_homo2[...,3:]

    # out shape: (b v) (h w) c -> b (v h w) c'
    world_xyz2 = einops.rearrange(world_xyz2, '(b v) n c -> b (v n) c', b=B, v=V)
    
    return world_xyz2


## init opt as global
opt = tyro.cli(AllConfigs)


# process the loaded splatters into 3-channel images
gt_attr_keys = ['pos', 'opacity', 'scale', 'rotation', 'rgbs']
start_indices = [0, 3, 4, 7, 11]
end_indices = [3, 4, 7, 11, 14]
attr_map = {key: (si, ei) for key, si, ei in zip (gt_attr_keys, start_indices, end_indices)}
sp_min_max_dict = {"z-depth": (0, 3),  # FIXME: should be 0-3
                   "xy-offset": (-0.7, 0.7), 
                   "xyz-offset": (-0.7, 0.7), 
                   "pos": (-0.7, 0.7), 
                   "scale": (-10., -2.),
                   }
ordered_attr_list = ["pos", # 0-3
                'opacity', # 3-4
                'scale', # 4-7
                "rotation", # 7-11
                "rgbs", # 11-14
            ] # must be an ordered list according to the channels

group_scale = False


if not group_scale:
    mode = opt.attr_group_mode # "v5" # "v3": for fake init
    if mode == "v1":
        attr_map.update({
            'scale-x': (4,5),
            'scale-y': (5,6),
            'scale-z': (6,7),
            "z-depth": (2,3),
            "xy-offset": (0,2), # TODO: actually can directly take xyz during real training. Now for vis purpose only
        })

        attr_cannot_be_encoded = ["scale", "pos"]
        for key in attr_cannot_be_encoded:
            del attr_map[key]

        ordered_attr_list = ["xy-offset", "z-depth", # 0-3
                'opacity', # 3-4
                'scale-x', 'scale-y', 'scale-z', # 4-7
                "rotation", # 7-11
                "rgbs", # 11-14
                ] # must be an ordered list according to the channels
       
    elif mode == "v2":
        attr_map.update({
            "z-depth": (2,3),
            "xy-offset": (0,2), # TODO: actually can directly take xyz during real training. Now for vis purpose only
        })

        attr_cannot_be_encoded = ["pos"]
        for key in attr_cannot_be_encoded:
            del attr_map[key]
    
        ordered_attr_list = ["xy-offset", "z-depth", # 0-3
                'opacity', # 3-4
                'scale', # 4-7
                "rotation", # 7-11
                "rgbs", # 11-14
                ] # must be an ordered list according to the channels
        
    elif mode in ["v3","v4"]:
        attr_map.update({
            "z-depth": (14,15), # TODO: find correct indices to insert this deptht
            "xyz-offset": (0,3), 
        })

        attr_cannot_be_encoded = ["pos"]
        for key in attr_cannot_be_encoded:
            del attr_map[key]
    
        ordered_attr_list = ["xyz-offset", # 0-3
                            "z-depth", # 14,15
                'opacity', # 3-4
                'scale', # 4-7
                "rotation", # 7-11
                "rgbs", # 11-14
                ] # must be an ordered list according to the channels
        ## INI lgm model to get xyz from depth
    elif mode in ["v5"]:
        pass
    else:
        raise ValueError ("Invalid ungrouping mode for splatter attributes")

else:
   
    # attr_map.update({
        # 'scale-x': (4,5),
        # 'scale-y': (5,6),
        # 'scale-z': (6,7),
        # "z-depth": (2,3),
        # "xy-offset": (0,2), # actually can directly take xyz during real training. Now for vis purpose only
    # })

    # attr_cannot_be_encoded = ["pos"]
    # for key in attr_cannot_be_encoded:
    #     del attr_map[key]
    pass

# assert # TODO: add check to cover each channel exactly once

print(f"Please confirm this attr to encode:\n{attr_map}")

def prepare_3channel_images_to_encode(splatters_mv):
    splatter_3Channel_image = {}

    for attr_to_encode, (start_i, end_i) in attr_map.items():
    
        splatter_attr = splatters_mv[:, start_i:end_i,...]

        sp_min, sp_max = None, None

        # process the channels
        if end_i - start_i == 1:
            print(f"repeat attr {attr_to_encode} for 3 times")
            splatter_attr = splatter_attr.repeat(1, 3, 1, 1) # [0,1]
        elif end_i - start_i == 3:
            pass
        elif attr_to_encode == "xy-offset":
            # ## normalize to [0,1]
            # sp_min, sp_max =  -1., 1.
            # splatter_attr = (splatter_attr - sp_min) / (sp_max - sp_min)
            ## cat one more dim
            # splatter_attr = torch.cat((splatter_attr, 0.5 * torch.ones_like(splatter_attr[:,0:1,...])), dim=1)
            splatter_attr = torch.cat((splatter_attr, torch.zeros_like(splatter_attr[:,0:1,...])), dim=1)
        elif attr_to_encode == "rotation":
            # st() # assert 4 is on the last dim
            # quaternion to axis angle
            quat = einops.rearrange(splatter_attr, 'b c h w -> b h w c')
            axis_angle = quaternion_to_axis_angle(quat)
            splatter_attr = einops.rearrange(axis_angle, 'b h w c -> b c h w')
            # st()

        else:
            raise ValueError(f"The dimension of {attr_to_encode} is problematic to encode")

        if "scale" in attr_to_encode:
            # use log scale
            splatter_attr = torch.log(splatter_attr)
            
            print(f"{attr_to_encode} log min={splatter_attr.min()} max={splatter_attr.max()}")
            sp_min, sp_max =  -10., -2.
            splatter_attr = (splatter_attr - sp_min) / (sp_max - sp_min) # [0,1]
            splatter_attr = splatter_attr.clip(0,1)

        elif attr_to_encode in ["z-depth", "xy-offset", "pos"] :
            # sp_min, sp_max =  splatter_attr.min(), splatter_attr.max()
            # sp_min, sp_max =  -1., 1.
            sp_min, sp_max =  -0.7, 0.7
            splatter_attr = (splatter_attr - sp_min) / (sp_max - sp_min)


        print(f"[{attr_to_encode}] in [0,1]: min={splatter_attr.min()} max={splatter_attr.max()}")

        sp_image = splatter_attr * 2 - 1 # [map to range [-1,1]]
        print(f"[{attr_to_encode}] in [-1, 1]: min={sp_image.min()} max={sp_image.max()}")
        print("[prepare_3channel_images_to_encode] finished")
       
        
        splatter_3Channel_image[attr_to_encode] = sp_image

    st()

    return splatter_3Channel_image


# from Marigold.easy_inference import MarigoldModel
# ## instantiate marigold model
# checkpoint_path = 'prs-eth/marigold-lcm-v1-0'
# marigold_model = MarigoldModel(checkpoint_path)





from torchvision import transforms
from PIL import Image
import cv2

def load_splatter_3channel_images_to_encode(splatter_dir, suffix="to_encode"):
    # valid siffices: ["decoded", "to_encode"]
    splatter_3Channel_image = {}
    
    for attr in ordered_attr_list:
        im_path = os.path.join(splatter_dir, f"{attr}_{suffix}.png")
        
        # image = cv2.imread(im_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255 # [512, 512, 4] in [0, 1]
        # image = torch.from_numpy(image)
        
        # Define a transform to convert the image to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to tensor and scales to [0, 1]
        ])
        
        image = Image.open(im_path).convert('RGB')  # Convert to RGB
        # Apply the transform to the image
        image = transform(image)
        image = einops.rearrange(image, "c h w -> h w c")
        
        print("images shape: ", image.shape) # (384, 256, 3)
        
        
        si, ei = attr_map[attr]
        if (ei - si) == 1:
            image = torch.mean(image, dim=-1, keepdim=True)
            image = image.repeat(1,1,3)
        elif attr == "rotation": 
            pass
            # ag = image[None] # einops.rearrange(, 'b c h w -> b h w c')
            # print("load axis angle as shape [b h w c]: ", ag.shape)
            # quaternion = axis_angle_to_quaternion(ag)
            # sp_image_o = einops.rearrange(quaternion, 'b h w c -> b c h w')
        else:
            assert (ei - si) == 3, print(f"{attr} has invalid si and ei: {si, ei}")
        
        splatter_3Channel_image[attr] = image  # [0,1]
    
     # reshape, [0,1] -> [-1,1]
    for key, value in splatter_3Channel_image.items():
        ## do reshapeing to [1, 3, 384, 256])
        new_value = einops.rearrange(value, "h w c -> c h w")[None]
        # print("new shape: ", new_value.shape)
        ## do mapping of value from [0,1] to [-1,1]
        if value.min() < 0 or value.max()>1:
            st()
        new_value = new_value * 2 - 1
        splatter_3Channel_image.update({key: new_value})
    
    depth_min, depth_max = 0, 3 
    sp_min_max_dict["z-depth"] = depth_min, depth_max 
    
    assert set(ordered_attr_list) == set(splatter_3Channel_image.keys())
    
    return splatter_3Channel_image
   
def load_splatter_png_as_original_channel_images_to_encode(splatter_dir, suffix="to_encode", device="cuda", ext="png"):
    # valid siffices: ["decoded", "to_encode"]
    print(f"Loading {suffix}_{ext} files")
    # NOTE: since we are loading png not ply, no need to do deactivation
    splatter_3Channel_image = {}
    
    for attr in ordered_attr_list:
        # im_path = os.path.join(splatter_dir, f"{attr}_{suffix}.png")
        im_path = os.path.join(splatter_dir, f"{attr}_{suffix}.{ext}")
        
        # image = cv2.imread(im_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255 # [512, 512, 4] in [0, 1]
        # image = torch.from_numpy(image)
        
        if ext in ["png"]:
            # Define a transform to convert the image to tensor
            transform = transforms.Compose([
                transforms.ToTensor(),  # Converts to tensor and scales to [0, 1]
            ])
            
            image = Image.open(im_path).convert('RGB')  # Convert to RGB
            # Apply the transform to the image
            image = transform(image)
            
        elif ext in ["pt"]:
            image = torch.load(im_path).detach().clone()
            image = (image + 1) * 0.5
            print(image.requires_grad, image.grad)
            
        else:
            print("Invalid extension choice: ", ext)
            
        image = einops.rearrange(image, "c h w -> h w c")
        print("images shape: ", image.shape) # (384, 256, 3)
        
        
        si, ei = attr_map[attr]
        if (ei - si) == 1:
            image = torch.mean(image, dim=-1, keepdim=True)
            # image = image.repeat(1,1,3) # keep the original channel, no repeat
        elif attr == "rotation": 
            pass
            # ag = image[None] # einops.rearrange(, 'b c h w -> b h w c')
            # print("load axis angle as shape [b h w c]: ", ag.shape)
            # quaternion = axis_angle_to_quaternion(ag)
            # sp_image_o = einops.rearrange(quaternion, 'b h w c -> b c h w')
        else:
            assert (ei - si) == 3, print(f"{attr} has invalid si and ei: {si, ei}")
        
        splatter_3Channel_image[attr] = image  # [0,1]
    
     # reshape, [0,1] -> [-1,1]
    for key, value in splatter_3Channel_image.items():
        ## do reshapeing to [1, 3, 384, 256])
        new_value = einops.rearrange(value, "h w c -> c h w")[None]
        print("new shape: ", new_value.shape)
        print(new_value.min(), new_value.max())
        ## do mapping of value from [0,1] to [-1,1]
        if value.min() < 0 or value.max()>1:
            # st()
            assert ext == "pt" # FIXME: change this back
            value = torch.clip(value, 0, 1)
            print("Clipping the value of pt tensor")
            # fixme: for debug only
            value = (value*255).to(torch.uint8).to(torch.float32) / 255
            print("Clipping the value to uint8")
        new_value = new_value * 2 - 1
        splatter_3Channel_image.update({key: new_value.to(device)})
    
    depth_min, depth_max = 0, 3 
    sp_min_max_dict["z-depth"] = depth_min, depth_max 

    
    
    assert set(ordered_attr_list) == set(splatter_3Channel_image.keys())
    
    return splatter_3Channel_image


# if opt.data_mode == "srn_cars":
#     global_bg_color = torch.ones(3, dtype=torch.float32, device="cuda")
#     assert not opt.color_augmentation
# else:
#     global_bg_color = torch.ones(3, dtype=torch.float32, device="cuda") * 0.5
    
def prepare_fake_original_channel_images_to_encode(data, splatters_mv, depth_mode=None, lgm_model=None, gs=None, opt=None, bg_color=None):
    # splatters_mv.shape -> torch.Size([1, 14, 384, 256])
    splatter_3Channel_image = {}

    ## TODO: important, do resize of fake images to 128 first!!!!
    ## FIXME: why? Since it is fake, why do we stick to 128? On the contrary, 320 is better since it is aligend with the original zero123++ bg

    ## rgb
    # resized_rgb = F.interpolate(data["input"][0], (128, 128))
    resized_rgb = data["input"][0]
    mv_rgb_images = einops.rearrange(resized_rgb, "(m n) c h w -> (m h) (n w) c", m=3, n=2) # [B, V, 3, output_size, output_size]
    kiui.write_image('data_input_rgb.jpg', mv_rgb_images.detach().cpu().numpy())
    splatter_3Channel_image["rgbs"] = mv_rgb_images # [0,1]

    
    ## opacity
    # cannot init as constant for all pixels!!!! Use alpha mask from gt data mask
    # resized_alpha = F.interpolate(data["masks_input"][0], (128, 128))
    resized_alpha = data["masks_input"][0]
    mv_alpha_images = einops.rearrange(resized_alpha, "(m n) c h w -> (m h) (n w) c", m=3, n=2) # [B, V, 3, output_size, output_size]
    kiui.write_image('data_input_mask.jpg', mv_alpha_images.detach().cpu().numpy())
    # splatter_3Channel_image["opacity"] = torch.cat([mv_alpha_images]*3, dim=-1) # [0,1]
    splatter_3Channel_image["opacity"] = mv_alpha_images # [0,1]
    kiui.write_image('data_fake_opacity_3channel.jpg', splatter_3Channel_image["opacity"])


    ## z-depth
    # TODO: infer the depth for each view independently? Using metric depth?
    ## get from marigold
    # Assume rgb_image is a PIL Image or a tensor with shape [H, W, 3]
    original_size_rgb = einops.rearrange(data["input"][0], "(m n) c h w -> (m h) (n w) c", m=3, n=2)
    depth_mode = "gaussian_rendering"
    if depth_mode=="marigold":
        depth_map_np = marigold_model.get_depth_from_image(original_size_rgb) # (768, 512). regardless of input shape, [0,1]
        # kiui.write_image('data_marigold_depth.jpg', depth_map_np)

        depth_map_mv = einops.rearrange(torch.from_numpy(depth_map_np), '(m h) (n w) -> (m n) h w', m=3, n=2)
        depth_map_mv = depth_map_mv.unsqueeze(1) # [V C H W]
        resized_depth = F.interpolate(depth_map_mv, (128, 128))
        reshaped_depth = einops.rearrange(resized_depth, "(m n) c h w -> (m h) (n w) c", m=3, n=2)# H, W, 1
        # splatter_3Channel_image["z-depth"] =  torch.cat([reshaped_depth]*3, dim=-1)
        splatter_3Channel_image["z-depth"] =  reshaped_depth
    elif depth_mode=="gaussian_rendering":
        # assert model is not None
        gaussians = fuse_splatters(data["splatters_output"])
        # bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device) * 0.5
        assert opt.render_input_views
        gs_results = gs.render(gaussians, data['cam_view'][:,:opt.num_input_views], 
                                     data['cam_view_proj'][:,:opt.num_input_views], data['cam_pos'][:,:opt.num_input_views], bg_color=bg_color)
                
        gt_depth = gs_results["depth"] # [1, V, 1, output_size, output_size]
        ##
        check_depth = False
        if check_depth:
            B, V, C, H, W= gt_depth.shape

            xyz_offset = torch.zeros_like(gt_depth.repeat(1,1,3,1,1))
            # world_xyz2 = lgm_model.get_world_xyz_from_depth_offset(gt_depth, xyz_offset, data=data)
            world_xyz2 = get_world_xyz_from_depth_offset(gt_depth, xyz_offset, data=data, opt=opt)
            pos = einops.rearrange(world_xyz2, 'b (m n h w) c -> b c (m h) (n w)', b=B, m=3, n=2, h=H, w=W)

            # save point cloud for debug
            save_pc = False
            save_mask_selected = True
            if save_pc:
                pc_dir = "debug_depth_pos_to_encode"
                os.makedirs(pc_dir, exist_ok=True)
                mv_pc = einops.rearrange(world_xyz2[0], '(v n) c -> v n c', v=V)
                if save_mask_selected:
                    mv_mask = einops.rearrange(resized_alpha, "v c h w -> v (h w) c") # v c h w 
                    mv_depth = einops.rearrange(gt_depth[0], "v c h w -> v (h w) c")
                    
                              
                for i, _v_pc in enumerate(mv_pc):
                    if save_mask_selected:
                        # save only visible by mask selection
                        # _v_mask = (mv_mask[i] > 0).squeeze(-1) 
                        _v_mask = (mv_depth[i] > 0).squeeze(-1)
                        print(torch.unique(_v_mask), _v_mask.shape)
                        _v_pc = _v_pc[_v_mask]
                        print(_v_pc.shape)
                        
                    save_as_ply(_v_pc, f"{pc_dir}/view_{i}.ply")
             
                print(f"Saved pc from depths to {pc_dir} ")
                st()
                

        # NOTE: this normalization is important to get clean depth map, 
        # but may lose information about the actual depth value used for view consistent recon
        depth_min, depth_max = 0, 3 
        sp_min_max_dict["z-depth"] = depth_min, depth_max 
        # st()
        
        gt_depth = (gt_depth - depth_min) / (depth_max - depth_min)

        gt_masks = gt_depth
        gt_masks_save = gt_masks.detach().cpu().numpy().transpose(0, 3, 1, 4, 2).reshape(-1, gt_masks.shape[1] * gt_masks.shape[3], 1) # [B*output_size, V*output_size, 3]
        kiui.write_image(f'diff_gs_w_depth.jpg', gt_masks_save)

        reshaped_depth = einops.rearrange(gt_depth[0], "(m n) c h w -> (m h) (n w) c", m=3, n=2)
        # splatter_3Channel_image["z-depth"] =  torch.cat([reshaped_depth]*3, dim=-1)
        splatter_3Channel_image["z-depth"] =  reshaped_depth


    else:
        print("Not a valid depth mode in prepare_fake_3channel_images_to_encode")
        exit()

    kiui.write_image('data_fake_depth_3channels.jpg', splatter_3Channel_image["z-depth"])

    ## xy offset # assume in [0,1]
    xyz_offset_zeros = torch.zeros_like(mv_rgb_images)
    offset_min, offset_max = sp_min_max_dict["xyz-offset"]
    xyz_offset_zeros = (xyz_offset_zeros - offset_min) / (offset_max - offset_min)
    splatter_3Channel_image["xyz-offset"] = xyz_offset_zeros


    ## scale
    # constant: 
    scale_constant = -5.7 # between -5.3 and -6
    # fake_scale_attr = torch.ones_like(mv_rgb_images) * torch.exp(
    #     torch.tensor([scale_constant], device=splatters_mv.device)
    #     )
    fake_scale_attr = torch.ones_like(mv_rgb_images) * scale_constant # already in log scale
    # map to [0,1]
    sp_min, sp_max = sp_min_max_dict["scale"]
    splatter_3Channel_image["scale"] = (fake_scale_attr - sp_min) / (sp_max - sp_min)
    

    ## rotation # assume in [0,1]
    # decoded_attr_image_dict[attr_to_encode] = F.normalize(torch.ones_like(splatter_attr). dim=-1)
    splatter_3Channel_image["rotation"] = torch.zeros_like(mv_rgb_images) # This is better than the above!! It's totally fine if all zeros
    # NOTE: this is already in axis angle since it is 3-channel
    
    # reshape, [0,1] -> [-1,1]
    for key, value in splatter_3Channel_image.items():
        ## do reshapeing to [1, 3, 384, 256])
        new_value = einops.rearrange(value, "h w c -> c h w")[None]
        print("new shape: ", new_value.shape)
        ## do mapping of value from [0,1] to [-1,1]
        if value.min() < 0 or value.max()>1:
            st()
        new_value = new_value * 2 - 1
        splatter_3Channel_image.update({key: new_value})
    
    
    print(set(ordered_attr_list))
    print(set(splatter_3Channel_image.keys()))
    assert set(ordered_attr_list) == set(splatter_3Channel_image.keys())


    return splatter_3Channel_image

def prepare_LGM_init_original_channel_images_to_encode(data, splatters_mv, depth_mode=None, lgm_model=None, gs=None, opt=None):
    # splatters_mv.shape -> torch.Size([1, 14, 384, 256])
    splatter_original_Channel_image = {}
    
    # get each attr from the splatters_mv

    # rgb
    for attr, (si, ei) in attr_map.items():
        # get correct channel
        splatter_im_activated = splatters_mv[:,si:ei,...]
        print("splatter_im_activated: ", splatter_im_activated.shape)
        
        # deactivate 
        if attr == "scale":
            splatter_im_deact = torch.log(splatter_im_activated)

        # other processing 
        elif attr == "rotation":
            quat = einops.rearrange(splatter_im_activated, 'b c h w -> b h w c')
            axis_angle = quaternion_to_axis_angle(quat)
            splatter_im_deact = einops.rearrange(axis_angle, 'b h w c -> b c h w')
        
        else:
            splatter_im_deact = splatter_im_activated
        
        # normalize
        if attr in ["rotation", "rgbs", "opacity"]:
            pass # because rgb and opacity are natually in [0,1]
        else:
            print("Normalizing ", attr)
            sp_min, sp_max = sp_min_max_dict[attr]
            splatter_im_deact = (splatter_im_deact - sp_min) / (sp_max - sp_min)
            
        # [0,1] -> [-1,1]
        splatter_im_deact = splatter_im_deact * 2 - 1

        splatter_original_Channel_image[attr] = splatter_im_deact
        print("splatter_im_deact: ", splatter_im_deact.shape)
        
    assert set(ordered_attr_list) == set(splatter_original_Channel_image.keys())

    return splatter_original_Channel_image


def original_to_3Channel_splatter(splatter_original_channels):
    
    splatter_3_channels = {}
    for key, value in splatter_original_channels.items():
        if value.shape[1] == 1:
            splatter_3_channels[key] = value.repeat(1,3,1,1)
            # print(f"{key} becomes: ", splatter_3_channels[key].shape)
            # print("[original_to_3Channel_splatter] range: ", value.min(), value.max())
        else:
            assert value.shape[1] == 3
            splatter_3_channels[key] = value
    
    return splatter_3_channels
    
    


def encode_3channel_image_to_latents(pipe, sp_image):
    
    # if not os.path.exists(os.path.join(output_path, name)):
    #     os.makedirs(os.path.join(output_path, name), exist_ok=True)
    # mv_image = einops.rearrange((sp_image[0].clip(-1,1)+1).cpu().numpy()*127.5, 'c h w-> h w c').astype(np.uint8) 
    # Image.fromarray(mv_image).save(os.path.join(output_path, f'{name}/{attr_to_encode}_to_encode.png'))

    # encode: splatter attr -> latent 
    # sp_image_original = sp_image.clone()
    sp_image = scale_image(sp_image.to(pipe.device))
    # st()
    sp_image = pipe.vae.encode(sp_image).latent_dist.sample() * pipe.vae.config.scaling_factor
    latents = scale_latents(sp_image)

    return latents


def decode_single_latents(pipe, latents, attr_to_encode, mv_image=None):
    start_i, end_i = attr_map[attr_to_encode]
    # print("decode_single_latents -> latents.requires_grad", latents.requires_grad)

    if mv_image==None: # else, just get original values from [-1,1]
        #  decode: latents -> platter attr
        latents1 = unscale_latents(latents)
        image = pipe.vae.decode(latents1 / pipe.vae.config.scaling_factor, return_dict=False)[0]
        image = unscale_image(image)
        # print("decode_single_latents", latents1.shape, " ->", image.shape)

        # print("decode_single_latents -> image.requires_grad", image.requires_grad)


        # # save decoded image
        # mv_image_numpy = einops.rearrange((image[0].clip(-1,1)+1).detach().cpu().numpy()*127.5, 'c h w-> h w c').astype(np.uint8) 
        # Image.fromarray(mv_image_numpy).save(os.path.join(output_path, f'{name}/{attr_to_encode}_pred.png'))

        # scale back to original range
        mv_image = image # [b c h w], in [-1,1] # return for visualization
        # print(f"[{attr_to_encode}] in [-1,1] : min={mv_image.min()} max={mv_image.max()}")

    # if attr_to_encode in[ "z-depth", "xy-offset"]:
    #     sp_image_o = mv_image
    #     # st() # NOTE: try clip z-depth? No need, they are already within the range
    # else:
    sp_image_o = 0.5 * (mv_image + 1) # [map to range [0,1]]

    
    # print(f"Decoded attr [0,1] {attr_to_encode}: min={sp_image_o.min()} max={sp_image_o.max()}")

    if "scale" in attr_to_encode:
        # v2
        sp_min, sp_max = sp_min_max_dict["scale"]

        sp_image_o = sp_image_o.clip(0,1) 
        sp_image_o = sp_image_o * (sp_max - sp_min) + sp_min
        
        sp_image_o = torch.exp(sp_image_o)
        # print(f"Decoded attr [unscaled] {attr_to_encode}: min={sp_image_o.min()} max={sp_image_o.max()}")

    # elif attr_to_encode == "z-depth":
        # sp_image_o = sp_image_o * (sp_max - sp_min) + sp_min

    elif attr_to_encode in[ "z-depth", "xy-offset", "pos", "xyz-offset"]:
        sp_min, sp_max = sp_min_max_dict[attr_to_encode]
        sp_image_o = sp_image_o * (sp_max - sp_min) + sp_min
        sp_image_o = torch.clamp(sp_image_o, min=sp_min, max=sp_max)
        # print(f"{attr_to_encode}: {sp_min, sp_max}")
        # st()
    
    

    if attr_to_encode == "xy-offset": 
        sp_image_o = sp_image_o[:,:2] # sp_image_o: torch.Size([1, 3, 384, 256])

    if attr_to_encode == "rotation": 
        
        ag = einops.rearrange(sp_image_o, 'b c h w -> b h w c')
        quaternion = axis_angle_to_quaternion(ag)
        sp_image_o = einops.rearrange(quaternion, 'b h w c -> b c h w')
        # st()

    if end_i - start_i == 1:
        # print(torch.allclose(torch.mean(sp_image_o, dim=1, keepdim=True), sp_image_o))
        # st()
        # print(f"Decoded attr [unscaled, before mean] {attr_to_encode}: min={sp_image_o.min()} max={sp_image_o.max()}")
        sp_image_o = torch.mean(sp_image_o, dim=1, keepdim=True) # avg.
        # print(f"Decoded attr [unscaled, after median] {attr_to_encode}: min={sp_image_o.min()} max={sp_image_o.max()}")
        
  
    # print(f"Decoded attr [unscaled] {attr_to_encode}: min={sp_image_o.min()} max={sp_image_o.max()}")
    
    return sp_image_o, mv_image

def save_as_ply(points, filename='output.ply'):
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
end_header
"""
    with open(filename, 'w') as f:
        f.write(header)
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")

# # Example tensor of shape [n, 3]
# points = torch.rand(10, 3)  # 10 random 3D points
# save_as_ply(points, 'points.ply')

def get_splatter_images_from_decoded_dict(decoded_attr_image_dict, lgm_model=None, data=None, group_scale=False, save_pc=False):
        
    # if not group_scale:
    #     ordered_attr_list = ["xy-offset", "z-depth", # 0-3
    #                         'opacity', # 3-4
    #                         'scale-x', 'scale-y', 'scale-z', # 4-7
    #                         "rotation", # 7-11
    #                         "rgbs", # 11-14
    #                         ] # must be an ordered list according to the channels
    # else:
    #     ordered_attr_list = ["pos", # 0-3
    #                         'opacity', # 3-4
    #                         'scale', # 4-7
    #                         "rotation", # 7-11
    #                         "rgbs", # 11-14
    #                         ] # must be an ordered list according to the channels
    
    if mode == "v3":
        assert lgm_model is not None and data is not None
        # has to convert the depth + offset to true pos
        depth_activated = decoded_attr_image_dict["z-depth"] # torch.Size([1, 98304, 1])
        B, C, _H, _W = depth_activated.shape
        assert _H / 3 == _W / 2 == 128
        V, H, W = 6, 128, 128

        print("depth decoded in original range: ", depth_activated.min(), depth_activated.max())
        # depth_activated = einops.rearrange(depth_activated, "b c _h _w -> b (_h _w) c")
        depth_activated = einops.rearrange(depth_activated, "b c (m h) (n w) -> b (m n) c h w", h=H, w=W, m=3, n=2)
        xyz_offset = decoded_attr_image_dict["xyz-offset"]
        # xyz_offset = einops.rearrange(xyz_offset, "b c _h _w -> b (_h _w) c")
        xyz_offset = einops.rearrange(xyz_offset, "b c (m h) (n w) -> b (m n) c h w", h=H, w=W, m=3, n=2)

        # world_xyz2 = lgm_model.get_world_xyz_from_depth_offset(depth_activated, xyz_offset, data=data)
        world_xyz2 = get_world_xyz_from_depth_offset(depth_activated, xyz_offset, data=data, opt=opt)
        pos = einops.rearrange(world_xyz2, 'b (m n h w) c -> b c (m h) (n w)', b=B, m=3, n=2, h=H, w=W)
        
        # save point cloud for debug
        # save_pc = True
        if save_pc:
            pc_dir = "debug_depth_pos"
            os.makedirs(pc_dir, exist_ok=True)
            mv_pc = einops.rearrange(world_xyz2[0], '(v n) c -> v n c', v=V)
            for i, _v_pc in enumerate(mv_pc):
                save_as_ply(_v_pc, f"{pc_dir}/view_{i}.ply")
            print(f"Saved pc from depths to {pc_dir} ")
            # st()
    # if mode == "v4":
    #     assert lgm_model is not None and data is not None
    #     # has to convert the depth + offset to true pos
    #     depth_activated = decoded_attr_image_dict["z-depth"] # torch.Size([1, 98304, 1])
    #     B, C, _H, _W = depth_activated.shape
    #     assert _H / 3 == _W / 2 == 320
    #     V, H, W = 6, 320, 320

    #     print("depth decoded in original range: ", depth_activated.min(), depth_activated.max())
    #     # depth_activated = einops.rearrange(depth_activated, "b c _h _w -> b (_h _w) c")
    #     depth_activated = einops.rearrange(depth_activated, "b c (m h) (n w) -> b (m n) c h w", h=H, w=W, m=3, n=2)
    #     xyz_offset = decoded_attr_image_dict["xyz-offset"]
    #     # xyz_offset = einops.rearrange(xyz_offset, "b c _h _w -> b (_h _w) c")
    #     xyz_offset = einops.rearrange(xyz_offset, "b c (m h) (n w) -> b (m n) c h w", h=H, w=W, m=3, n=2)

    #     print(f"decoded depth_activated: {depth_activated.shape}, xyz_offset: {xyz_offset.shape}")
    #     world_xyz2 = lgm_model.get_world_xyz_from_depth_offset(depth_activated, xyz_offset, data=data)
    #     pos = einops.rearrange(world_xyz2, 'b (m n h w) c -> b c (m h) (n w)', b=B, m=3, n=2, h=H, w=W)
        
    #     # save point cloud for debug
    #     save_pc = True
    #     if save_pc:
    #         pc_dir = "debug_depth_pos"
    #         os.makedirs(pc_dir, exist_ok=True)
    #         mv_pc = einops.rearrange(world_xyz2[0], '(v n) c -> v n c', v=V)
    #         for i, _v_pc in enumerate(mv_pc):
    #             save_as_ply(_v_pc, f"{pc_dir}/view_{i}.ply")
    #         print(f"Saved pc from depths to {pc_dir} ")
    #         st()
            
            
        ## the order should also change, not the same as the ordered list since xyz-depth channnel mismatch
        attr_image_list = [pos] 
        attr_image_list += [decoded_attr_image_dict[attr] for attr in ordered_attr_list[2:] ]

    else:

        attr_image_list = [decoded_attr_image_dict[attr] for attr in ordered_attr_list ]
        # [print(t.shape) for t in attr_image_list]

    splatter_mv = torch.cat(attr_image_list, dim=1)

    ## reshape 
    splatters_to_render = einops.rearrange(splatter_mv, 'b c (h2 h) (w2 w) -> b (h2 w2) c h w', h2=3, w2=2) 
    
    return splatters_to_render
                    
def render_from_decoded_images(gs, splatters_to_render, data, bg_color):
    gaussians = fuse_splatters(splatters_to_render)
    # if opt.data_mode == "srn_cars":
    #     bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device) 
    # else:
    #     bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device) * 0.5
    gs_results = gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
                    
    return gs_results

from kiui.lpips import LPIPS
global_lpips_loss = LPIPS(net='vgg')
global_lpips_loss.requires_grad_(False)
global_lpips_loss.to("cuda")

def calculate_loss(model, gs_results, data, save_gt_path=None):
    results = {}
    loss = 0
    
    pred_images = gs_results['image'] # [B, V, C, output_size, output_size]
    pred_alphas = gs_results['alpha'] # [B, V, 1, output_size, output_size]

    gt_images = data['images_output'] # [B, V, 3, output_size, output_size], ground-truth novel views
    gt_masks = data['masks_output'] 

    if opt.data_mode == "srn_cars":
        gt_images = gt_images * gt_masks + (1 - gt_masks)# NOTE: white bg
    else:
        gt_images = gt_images * gt_masks + (1 - gt_masks) * opt.bg # NOTE: gray bg
    if save_gt_path!=None:
        gt_images_save = gt_images.detach().cpu().numpy().transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
        kiui.write_image(os.path.join(save_gt_path, 'calculate_loss_image_gt.jpg'), gt_images_save)


    if opt.data_mode == "srn_cars":
        loss_mse_rendering = F.mse_loss(pred_images, gt_images) 
    else:
        loss_mse_rendering = F.mse_loss(pred_images, gt_images) + F.mse_loss(pred_alphas, gt_masks)
    results['loss_rendering'] = loss_mse_rendering
    loss = loss + loss_mse_rendering

    if opt.lambda_lpips > 0:
        # loss_lpips = model.lpips_loss(
        loss_lpips = global_lpips_loss(
            F.interpolate(gt_images.view(-1, 3, opt.output_size, opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
            F.interpolate(pred_images.view(-1, 3, opt.output_size, opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
        ).mean()
        results['loss_lpips'] = loss_lpips
        loss = loss + opt.lambda_lpips * loss_lpips
  
        print(f"loss lpips:{loss_lpips}")
    
    results['loss'] = loss
    
    # metric
    with torch.no_grad():
        psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
        results['psnr'] = psnr
    
        
    return results

def save_3channel_splatter_images(splatter_dict, fpath, range_min=0, suffix="decoded"):
    # os.makedirs(os.path.join(output_path, f'{name}/{iteration}'), exist_ok=True)
    os.makedirs(fpath, exist_ok=True)

    save_tensor = True
    for attr_to_encode, image in splatter_dict.items():
        # save decoded image
        # if range_min==-1:
        if save_tensor:
            torch.save(image[0], os.path.join(fpath, f'{attr_to_encode}_{suffix}.pt'))

        image_array = (image[0].clip(-1,1)+1).detach().cpu().numpy()*127.5
        # else:
        #     image_array = (image[0].clip(0,1)).detach().cpu().numpy()*255

        if image_array.shape[0]==3:
            mv_image_numpy = einops.rearrange(image_array, 'c h w-> h w c').astype(np.uint8) 
      
        Image.fromarray(mv_image_numpy).save(os.path.join(fpath, f'{attr_to_encode}_{suffix}.png'))
        # print(f"[save_3channel_splatter_images] {attr_to_encode}-{suffix} ", image.mean())
        
        print(f"{attr_to_encode} saved!")
    # st()

def save_gs_rendered_images(gs_results, fpath):
    pred_images = gs_results['image'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
    kiui.write_image(os.path.join(fpath, 'gs_render_rgb.png'), pred_images)

    pred_alphas = gs_results['alpha'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
    pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
    kiui.write_image(os.path.join(fpath, 'gs_render_alpha.png'), pred_alphas)

import torch.nn as nn
class SimpleLinearModel(nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.linear(x)
    

def main():    
    import sys

    # # Your additional path
    # # your_path = "/home/xuyimeng/Repo/LGM"
    # your_path = " /home/chenwang/xuyi_runs"

    # # Add your path to sys.path
    # sys.path.append(your_path)
   

    # opt = tyro.cli(AllConfigs)
    
    if opt.set_random_seed:
        # Set a manual seed for reproducibility 
        seed = 42
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )

    # # model
    # if opt.model_type == 'Zero123PlusGaussian':
    #     model = Zero123PlusGaussian(opt)
    #     from core.dataset_v4_batch import ObjaverseDataset as Dataset
    # elif opt.model_type == 'Zero123PlusGaussianCode':
    #     model = Zero123PlusGaussianCode(opt)
    #     from core.dataset_v4_code import ObjaverseDataset as Dataset
    
    # elif opt.model_type == 'Zero123PlusGaussianCodeUnetLora':
    #     model = Zero123PlusGaussianCodeUnetLora(opt)
    #     from core.dataset_v4_code import ObjaverseDataset as Dataset
   
    

    # lgm_model = None
    # if mode in ["v3", "v4"]:
    from core.models_fix_pretrained_depth_offset import LGM
    opt.use_splatter_with_depth_offset = True
    lgm_model = LGM(opt)
    lgm_model.requires_grad_(False).eval()
    
    # model = lgm_model
    # model = SimpleLinearModel()
    from core.models_zero123plus_marigold_unet_rendering_loss_cross_domain import Zero123PlusGaussianMarigoldUnetCrossDomain, fuse_splatters
    model =  Zero123PlusGaussianMarigoldUnetCrossDomain(opt)
    if opt.data_mode == "srn_cars":
        from core.dataset_v4_code_srn import SrnCarsDataset as Dataset
    else:
        from core.dataset_v4_code import ObjaverseDataset as Dataset


    # Check the number of GPUs
    num_gpus = accelerator.num_processes
    if accelerator.is_main_process:
        print(f"Num gpus: {num_gpus}")
    if num_gpus <= 1:
        time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        # Pick output directory.
        prev_run_dirs = []
        outdir = opt.workspace
        if os.path.isdir(outdir):
            prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        time_str = f'{cur_run_id:05d}'
        accelerator.wait_for_everyone()
    
    # c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    # assert not os.path.exists(c.run_dir)

    
    # # gathered_info = accelerator.all_gather(info_to_share)
    # accelerator.wait_for_everyone()
    # time_str = ''.join(chr(int(item)) for item in time_tensor.tolist())
    # print(time_str)
    
    # # Use torch.distributed to broadcast the workspace to all processes
    # time_str = torch.tensor(time_str.encode(), dtype=torch.uint8)
    # dist.broadcast(workspace, 0)  # Assuming rank 0 is the main process

    # workspace = workspace.decode()
        
    loss_str = 'loss'
    assert (opt.lambda_rendering + opt.lambda_splatter + opt.lambda_lpips > 0), 'Must have at least one loss'
    if opt.lambda_rendering > 0:
        loss_str+=f'_render{opt.lambda_rendering}'
    elif opt.lambda_alpha > 0:
        loss_str+=f'_alpha{opt.lambda_alpha}'
    if opt.lambda_splatter > 0:
        loss_str+=f'_splatter{opt.lambda_splatter}'
    if opt.lambda_lpips > 0:
        loss_str+=f'_lpips{opt.lambda_lpips}'
   
    desc = opt.desc
    if opt.vae_on_splatter_image:
        desc += "-vae_on_splatter_image"
    if opt.codes_from_encoder:
        desc += "-codes_from_encoder"
    elif opt.codes_from_diffusion:
        desc += "-codes_from_diffusion"
    elif opt.codes_from_cache:
        desc += "-codes_from_cache"
    
    
    assert (opt.one_step_diffusion is None) or (opt.lipschitz_mode is None)
    if opt.one_step_diffusion is not None:
        desc += f"_ONE_STEP_T={opt.one_step_diffusion}"
    if opt.lipschitz_mode is not None:
        desc += f"_lipschitz_mode={opt.lipschitz_mode}_coeff={opt.lipschitz_coefficient}"
        
        
    if opt.resume_workspace is not None:
        opt.workspace = opt.resume_workspace
    else:
            
        opt.workspace = os.path.join(opt.workspace, f"{time_str}-{desc}-{loss_str}-lr{opt.lr}-{opt.lr_scheduler}")
        
    
        if accelerator.is_main_process:
            assert not os.path.exists(opt.workspace)
            print(f"makdir: {opt.workspace}")
            os.makedirs(opt.workspace, exist_ok=True)
            # writer = tensorboard.SummaryWriter(opt.workspace)
    
    # real_workspace = sorted(os.listdir(os.path.dirname(opt.workspace)))[-1]
    # opt.workspace = real_workspace
    print(f"workspace: {opt.workspace}")
    
    from core.gs_w_depth import GaussianRenderer
    gs = GaussianRenderer(opt=opt)

    # # broadcast the opt.workspace to all processes
    # workspace_tensor = torch.tensor(list(opt.workspace.encode()), device="cuda", dtype=torch.uint8)
    # workspace_info = {'workspace tensor': workspace_tensor}
    # # Broadcast the workspace_info dictionary
    # workspace_info = broadcast(workspace_info, from_process=0)

    # # Decode the workspace string from the tensor
    # opt.workspace = bytes(workspace_info['workspace tensor'].tolist()).decode()

    # # Convert workspace string to a tensor
    # workspace_tensor = torch.tensor(bytearray(opt.workspace, 'utf-8'), dtype=torch.uint8).to("cuda")
    # # Broadcast the tensor
    # broadcasted_workspace_tensor = broadcast(workspace_tensor)
    # # Decode the tensor back to a string
    # decoded_workspace = broadcasted_workspace_tensor.cpu().numpy().tobytes().decode('utf-8')

    # # Use the decoded workspace
    # opt.workspace = decoded_workspace
    # print(f"Decoded workspace: {opt.workspace}")
    
    # accelerator.wait_for_everyone() 
    
    if opt.codes_from_cache:
        if opt.code_cache_dir is not None:
            code_cache_dir = opt.code_cache_dir
        else:
            resume_dir = os.path.dirname(opt.resume)
            if "epoch" in os.path.basename(resume_dir):
                resume_dir = os.path.dirname(resume_dir)
            code_cache_dir = os.path.join(resume_dir, "code_dir")
            assert os.isdir(code_cache_dir)
        print(f"Codes cache are loaded from:{code_cache_dir}")

    if accelerator.is_main_process:
        src_snapshot_folder = os.path.join(opt.workspace, 'src')
        src_i = 1
        while os.path.exists(src_snapshot_folder):
            # resume folder
            # for i in range(1,100): # assume the number of resume does not pass 100
            src_snapshot_folder = os.path.join(opt.workspace, f'src_{src_i:03d}')
            src_i += 1
            # if not os.path.exists(src_snapshot_folder):
            #     if opt.verbose:
            #         print(f"Resume src folder: {src_snapshot_folder}")
            #     break
    
        ignore_func = lambda d, files: [f for f in files if f.endswith('__pycache__')]
        # for folder in ['core', 'scripts', 'zero123plus']:
        for folder in ['core', 'scripts']:
            dst_dir = os.path.join(src_snapshot_folder, folder)
            shutil.copytree(folder, dst_dir, ignore=ignore_func, dirs_exist_ok=True)
        # Define the pattern to search for
        pattern = "main_zero123plus_v4_batch_code_inference_marigold_*"
        # Use glob to find all files matching the pattern
        files = glob.glob(pattern)
        for file in files:
            # for file in ['main_zero123plus_v4_batch_code_inference_marigold_v2_fake_init.py']:
            dest_file = os.path.join(src_snapshot_folder, file)
            shutil.copy2(file, dest_file)
        
    # resume
    # resume = not opt.vae_on_splatter_image
    resume = True
    if resume:
        assert opt.resume is not None ## only for decoder
        print(f"Resume from ckpt: {opt.resume}")
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(opt.resume, device='cpu')
        else:
            ckpt = torch.load(opt.resume, map_location='cpu')
        
        # tolerant load (only load matching shapes)
        model.load_state_dict(ckpt, strict=False)
        state_dict = model.state_dict()
        for k, v in ckpt.items():
            # if "lora" in k:
            #     print(f"not loading: {k}")
            #     # continue
            if k in state_dict: 
                if state_dict[k].shape == v.shape:
                    state_dict[k].copy_(v)
                else:
                    accelerator.print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
            else:
                accelerator.print(f'[WARN] unexpected param {k}: {v.shape}')
            
        del state_dict[k]
        # st()
        model.requires_grad_(False).eval()
        
        # non_loaded_params = [name for name, _ in state_dict.items() if 'lora' not in name]
        # print(non_loaded_params)
        # print("non_loaded_params that are not in ckpt")
        # decoder_params = [name for name, _ in ckpt.items() if 'decoder' in name]
        # print(decoder_params)
        # print("decoder_params")
        # st()
    
    # ## also load pretrained unet
    # if opt.resume_unet is not 
        

    # No need to copy the code_dir: handled by load_scenes already.
    train_dataset = Dataset(opt, training=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        # shuffle=True,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_dataset = Dataset(opt, training=False, prepare_white_bg=True)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )


    # # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.05, betas=(0.9, 0.95))
   

    # scheduler (per-iteration)
    if opt.lr_scheduler == 'CosAnn':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3000, eta_min=1e-6)
    elif opt.lr_scheduler == 'OneCyc':
        total_steps = opt.num_epochs * len(train_dataloader)
        pct_start = 3000 / total_steps
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.lr, total_steps=total_steps, pct_start=pct_start)
    elif opt.lr_scheduler == 'Plat':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=opt.lr_scheduler_factor, patience=opt.lr_scheduler_patience, verbose=True, min_lr=opt.min_lr_scheduled)
    else:
        assert ValueError('Not a valid lr_scheduler option.')

    # accelerate
    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, scheduler
    )

            # eval
    # with torch.no_grad():
    if True:
        model.eval()

        total_loss = 0
        total_psnr = 0
        total_loss_splatter = 0 #torch.tensor([0]).to()
        total_loss_rendering = 0 #torch.tensor([0])
        total_loss_alpha = 0
        total_loss_lpips = 0

        
        if opt.codes_from_diffusion or opt.vae_on_splatter_image:
                    
            # # # Load the pipeline
            pipeline_0123 = DiffusionPipeline.from_pretrained(
                "sudo-ai/zero123plus-v1.1", custom_pipeline="/mnt/kostas-graid/sw/envs/chenwang/workspace/diffgan/training/modules/zero123plus.py",
                torch_dtype=torch.float32
            )
            pipeline_0123.to('cuda:0')
            pipeline_0123.vae.requires_grad_(False).eval()
            pipeline_0123.unet.requires_grad_(False).eval()
            # pipeline = model.pipe
            # pipeline_0123 = model.pipe
            # model.requires_grad_(False).eval()

            pipeline = pipeline_0123
            # print("pipeline 0123")
            # print(pipeline.unet) # check whether lora is here
            # st()

            # Feel free to tune the scheduler!
            # `timestep_spacing` parameter is not supported in older versions of `diffusers`    
            # so there may be performance degradations
            # We recommend using `diffusers==0.20.2`
            pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                pipeline.scheduler.config, timestep_spacing='trailing'
            )
            # pipeline.scheduler = DDIMScheduler.from_config(
            #     pipeline.scheduler.config
            # )
            pipeline.to('cuda:0')
            
            output_path = f"{opt.workspace}/zero123plus/outputs_v3_inference_my_decoder"
            
            pipeline.prepare()
            guidance_scale = 1.5
            # guidance_scale = 1


       

        
        print(f"Save to run dir: {opt.workspace}")
        
        # Skip to the start_index in the dataloader
        # data_iterator = islice(enumerate(test_dataloader), opt.scene_start_index, opt.scene_end_index)

        # for _i, data in data_iterator:
        for _i, data in enumerate(test_dataloader):
            
            i = _i + opt.scene_start_index
            print("Scene ", i)
        
            # if i == 0:
            #     continue
            # if i > 40:
            #     exit(0)
            
            # Parameters for early stopping
            # best_loss = float('inf')
            best_psnr =0
            patience_counter = 0
            patience_limit = 50  # You can adjust this value
            delta = 1e-2  # The threshold for improvement, can be a percentage of best_loss

           
            # exactly loop [scene_start_index, scene_end_index)
            if i  < opt.scene_start_index: 
                continue
                
            if i == opt.scene_end_index: 
                print(f"Finished scenes from [{opt.scene_start_index}, {opt.scene_end_index}) !! Exit ")
                exit(0)
          
            scene_name = data["scene_name"][0]
            # if i < 5 or scene_name == "0a9b36d36e904aee8b51e978a7c0acfd":
            #     pass
            # else:
            #     continue
            
            directory = f'{opt.workspace}/eval_ckpt/{accelerator.process_index}_{i}_{scene_name}'
            # if not os.path.exists(directory):
            #     os.makedirs(directory)
            

            if opt.vae_on_splatter_image:

                pipe = pipeline_0123

                print("---------- Begin vae_on_splatter_image ---------------")
                        
                
                # path =f'/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/{scene_name}/000.png'
                if opt.data_mode == "srn_cars":
                    path = f"{opt.data_path_rendering}/{scene_name}/rgb/000000.png"
                else:
                    path = f"{opt.data_path_rendering}/{scene_name}/000.png"
                    
                print("Cond path is :", path)
            
                if opt.data_mode == "srn_cars":
                    name = path.split('/')[-3]
                else:
                    name = path.split('/')[-2]
                name = f"{i}_{name}"

                # check whether

                def extract_first_number(folder_name):
                    match = re.search(r'\d+', folder_name)
                    return int(match.group()) if match else None

                def check_scene_finished(scene_workspace):
                    # print(scene_workspace)
                    # st()
                    if os.path.exists(scene_workspace):
                        # if opt.verbose:
                        print(f"Already exists {i}th scene")
                    else:
                        return False
                    

                    # scene_finished = False
                
                    for item in os.listdir(scene_workspace):
                        # if not item.startswith('eval'):
                        #     continue 
                        item_epoch = extract_first_number(item)
                        # if item_epoch is None or item.startswith('events'):
                        if item_epoch is None or not os.path.isdir(os.path.join(scene_workspace, item)):
                            continue

                        # print(f"extract first number from item {item}: ",extract_first_number(item))
                        # print(item)
                        print(extract_first_number(item))
                        if item.endswith('_success'):
                            if opt.verbose:
                                print(f"Already early stopped.")
                            return True
                         
                        elif extract_first_number(item)>=opt.num_epochs-1:# already achieved the max training epochs
                            if opt.verbose:
                                print(f"Already achieved the max training epochs.")
                            return True
            
                    # ---------
                
                
                scene_finished = check_scene_finished(os.path.join(output_path, name)) 
                if scene_finished:
                    # # st()
                    # if 
                    continue 
                    
                
                print("This scene needs to be processed ", i)
                # continue
                # st()
                
                os.makedirs(os.path.join(output_path, name), exist_ok=True)

                img = to_rgb_image(Image.open(path))
                
                img.save(os.path.join(output_path, f'{name}/cond.png'))
                cond = [img]
        
                splatters_mv = einops.rearrange(data["splatters_output"], 'b (h2 w2) c h w -> b c (h2 h) (w2 w)', h2=3, w2=2) 
                    
               
                
                # process each attr to be 3-channel images
                # TODO: instead of getting images from splatters_mv, instead, getting them from real rgb
                # splatter_3Channel_image_to_encode = prepare_3channel_images_to_encode(splatters_mv)
                # # st()

                if opt.data_mode == "srn_cars":
                    global_bg_color = torch.ones(3, dtype=torch.float32, device="cuda")
                    assert not opt.color_augmentation
                else:
                    # global_bg_color = torch.ones(3, dtype=torch.float32, device="cuda") * 0.5
                    global_bg_color = torch.ones(3, dtype=torch.float32, device="cuda")

                if opt.splatter_to_encode is not None:

                    ## combine the splatter_to_encode with the scene name
                    load_path = os.path.join(opt.splatter_to_encode, f"{name}/{opt.load_iter}")
                    print("Scene specific load path is : ", load_path)
                    
                    try:
                        splatter_original_Channel_image_to_encode = load_splatter_png_as_original_channel_images_to_encode(load_path, device=splatters_mv.device, suffix=opt.load_suffix, ext=opt.load_ext)
                    except FileNotFoundError:
                        # Handle the FileNotFoundError exception here
                        print("File not found:", load_path)
                        # Optionally, re-raise the exception to propagate it further
                        # raise
                        continue
                    except Exception as e:
                        # Handle other types of exceptions
                        print("An error occurred:", e)
                    # else:
                    #     # Code to execute if no exception is raised
                    #     # For example, reading from the file or performing other operations
                    #     print("File opened successfully")
                    # finally:
                    #     # Code to execute regardless of whether an exception occurred
                    #     print("Finally block")

                    ## deprecated usages: (explicit load suffix and ext)
                    # splatter_3Channel_image_to_encode = load_splatter_3channel_images_to_encode(opt.splatter_to_encode)
                    # splatter_original_Channel_image_to_encode = load_splatter_png_as_original_channel_images_to_encode(opt.splatter_to_encode, device=splatters_mv.device)
                    # splatter_original_Channel_image_to_encode = load_splatter_png_as_original_channel_images_to_encode(opt.splatter_to_encode, device=splatters_mv.device, ext="pt")
                    # splatter_original_Channel_image_to_encode = load_splatter_png_as_original_channel_images_to_encode(opt.splatter_to_encode, suffix="decoded", device=splatters_mv.device)
                    
                
                elif opt.attr_group_mode == "v5":
                    splatter_original_Channel_image_to_encode = prepare_LGM_init_original_channel_images_to_encode(data, splatters_mv, lgm_model=lgm_model, gs=gs, opt=opt)
                
                else:
                    splatter_original_Channel_image_to_encode = prepare_fake_original_channel_images_to_encode(data, splatters_mv, lgm_model=lgm_model, gs=gs, opt=opt, bg_color=global_bg_color)
                # NOTE: prepare_fake_3channel_images_to_encode returns value in [-1,1]
                
                
                
                # # doing resize 
                # for attr_to_encode, splatter_attr in splatter_original_Channel_image_to_encode.items():
                #     splatter_attr_mv = einops.rearrange(splatter_attr[0], "c (m h) (n w) -> (m n) c h w", m=3, n=2)
                #     splatter_attr_mv_resized = F.interpolate(splatter_attr_mv, size=(320, 320), mode='bilinear', align_corners=False)
                #     splatter_original_Channel_image_to_encode[attr_to_encode] = einops.rearrange(splatter_attr_mv_resized, "(m n) c h w -> c (m h) (n w)", m=3, n=2)[None]
                
                
                print()
                for attr_to_encode, splatter_attr in splatter_original_Channel_image_to_encode.items():
                    # print(attr_to_encode, splatter_attr.shape, splatter_attr.min(), splatter_attr.max())
                    splatter_attr.requires_grad_(True)

                
                optimizer = torch.optim.Adam([splatter_original_Channel_image_to_encode[attr] for attr in ordered_attr_list], lr=0.01)

                # for param_group in optimizer.param_groups:
                #     for param in param_group['params']:
                #         print(param.requires_grad, param.shape)


                # Optimization loop
                num_iterations = opt.num_epochs      
                save_iters = 1 # num_iterations // 10
                
                if opt.splatter_to_encode is not None or opt.resume is not None: #  and opt.attr_group_mode != "v5":
                    num_iterations = 1
                    
                acceptable_loss_threshold = 10
                decoded_3channel_attr_image_dict = {}
                decoded_attributes_dict = {}

                
                # save init splatter image before optimization
                print(f"output_path={output_path}, name={name}")
                splatter_3Channel_image_to_encode = original_to_3Channel_splatter(splatter_original_Channel_image_to_encode)
                save_3channel_splatter_images(splatter_3Channel_image_to_encode, fpath=os.path.join(output_path, f'{name}/init'), range_min=-1, suffix="encoded")
                # st()
                
                to_encode_attributes_dict_init = {}
                with torch.no_grad():
                    for attr, image in splatter_3Channel_image_to_encode.items():
                        decoded_attributes, _ = decode_single_latents(pipeline_0123, None, attr_to_encode=attr, mv_image=image)
                        to_encode_attributes_dict_init.update({attr:decoded_attributes}) # splatter attributes in original range
                
                with torch.no_grad():
                    splatters_to_render = get_splatter_images_from_decoded_dict(to_encode_attributes_dict_init, lgm_model=lgm_model, data=data, group_scale=group_scale)
                    # print("splatters_to_render.requires_grad: ", splatters_to_render.requires_grad)
                    # Render the decoded images using the splatter model
                    
                    # bg_color =  torch.ones(3, dtype=torch.float32, device=gaussians.device) * 0.5
                    gs_results = render_from_decoded_images(gs, splatters_to_render, data=data, bg_color=global_bg_color)
                    # save rendering results
                    save_gs_rendered_images(gs_results, fpath=os.path.join(output_path, f'{name}/init'))
                    

                gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                kiui.write_image(os.path.join(output_path, f'{name}/gt/image_gt.jpg'), gt_images)

                # save init rendering
                ## gt splatter image
                gaussians = fuse_splatters(data["splatters_output"])
                # bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device) * 0.5
                gs_results = gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=global_bg_color)
            
                save_gs_rendered_images(gs_results, fpath=os.path.join(output_path, f'{name}/gt'))

                writer = tensorboard.SummaryWriter(os.path.join(output_path, f'{name}'))

                # Define a shared text file to write PSNR values
                psnr_log_file = os.path.join(output_path, 'psnr_log.txt')
                                                   
                # for i in range(num_iterations):
                for i in tqdm(range(num_iterations), desc='Optimization Progress'):
                    
                    if opt.color_augmentation and  i % save_iters != 0:
                        global_bg_color = torch.rand(3, dtype=torch.float32, device=gaussians.device)
     
                    optimizer.zero_grad()  # Perform a si
                    
                    print(f"[Iter {i}]")
                    
                    
                    splatter_3Channel_image_to_encode = original_to_3Channel_splatter(splatter_original_Channel_image_to_encode)
                    
                    # vae.encode
                    latents_all_attr_dict = {}
                
                    # print(ordered_attr_list)
                    for attr in ordered_attr_list:
                        latents_single_attr = encode_3channel_image_to_latents(pipe, splatter_3Channel_image_to_encode[attr])
                        latents_all_attr_dict.update({attr: latents_single_attr})

                    latents_all_attr_tensor = torch.cat([latents_all_attr_dict[attr] for attr in ordered_attr_list])
                    # print(latents_all_attr_tensor.shape, len(ordered_attr_list)) # torch.Size([5, 4, 48, 32])
                    assert latents_all_attr_tensor.shape[0] == len(ordered_attr_list)
    
                    with torch.no_grad():
                        if opt.cd_spatial_concat or opt.custom_pipeline in ["./zero123plus/pipeline_v7_seq.py"]:
                            if opt.cd_spatial_concat:
                                gt_latents = einops.rearrange(latents_all_attr_tensor, "(B A) C (m H) (n W) -> B C (A H) (m n W)", B=data['cond'].shape[0], m=3, n=2)
                            elif opt.custom_pipeline in ["./zero123plus/pipeline_v7_seq.py"]:
                                gt_latents = latents_all_attr_tensor
                            else:
                                assert NotImplementedError
                                
                            with torch.no_grad():
                                # pipeline =  # .unet.requires_grad_(False).eval()
                                
                                prompt_embeds, cak = model.pipe.prepare_conditions(cond, guidance_scale=guidance_scale)
                                if opt.custom_pipeline in ["./zero123plus/pipeline_v7_seq.py"]:
                                    # print(procmpt_embeds.shape)
                                    prompt_embeds = torch.cat([prompt_embeds[0:1]]*gt_latents.shape[0] + [prompt_embeds[1:]]*gt_latents.shape[0], dim=0) # torch.Size([10, 77, 1024])
                                    cak['cond_lat'] = torch.cat([cak['cond_lat'][0:1]]*gt_latents.shape[0] + [cak['cond_lat'][1:]]*gt_latents.shape[0], dim=0)
                                    
                                print(f"cak: {cak['cond_lat'].shape}") # always 64x64, not affected by cond size
                                model.pipe.scheduler.set_timesteps(30, device='cuda:0')
                                # if opt.one_step_diffusion is not None:
                                #     pipeline.scheduler.set_timesteps(opt.one_step_diffusion, device='cuda:0')
                                    
                                timesteps = model.pipe.scheduler.timesteps
                                
                                debug = False
                                if debug:
                                    debug_t = torch.tensor(50, dtype=torch.int64, device='cuda:0',)
                                    noise = torch.randn_like(gt_latents, device='cuda:0', dtype=torch.float32)
                                    t = torch.ones((5,), device=gt_latents.device, dtype=torch.int)
                                    latents = model.pipe.scheduler.add_noise(gt_latents, noise, t*debug_t)
                                    
                                    timesteps = [debug_t]
                                else:
                                    latents  = torch.randn_like(gt_latents, device='cuda:0', dtype=torch.float32)
                                
                                domain_embeddings = torch.eye(5).to(latents.device)
                                if opt.cd_spatial_concat:
                                    domain_embeddings = torch.sum(domain_embeddings, dim=0, keepdims=True) # feed all domains
                                domain_embeddings = torch.cat([
                                        torch.sin(domain_embeddings),
                                        torch.cos(domain_embeddings)
                                    ], dim=-1)
                                
                            
                                # latents_init = latents.clone().detach()
                                for _, t in enumerate(timesteps):
                                    print(f"enumerate(timesteps) t={t}")
                            
                                    latent_model_input = torch.cat([latents] * 2)
                                    # domain_embeddings = torch.cat([domain_embeddings] * 2)
                                    latent_model_input = model.pipe.scheduler.scale_model_input(latent_model_input, t)

                                    # predict the noise residual
                                    noise_pred = model.unet(
                                        latent_model_input,
                                        t,
                                        encoder_hidden_states=prompt_embeds,
                                        cross_attention_kwargs=cak,
                                        return_dict=False,
                                        class_labels=domain_embeddings,
                        
                                    )[0]    

                                    # perform guidance
                                    if True:
                                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                                    # compute the previous noisy sample x_t -> x_t-1
                                    if debug:
                                        alphas_cumprod = model.pipe.scheduler.alphas_cumprod.to(
                                            device=latents.device, dtype=latents.dtype
                                        )
                                        alpha_t = (alphas_cumprod[t] ** 0.5).view(-1, 1, 1, 1)
                                        sigma_t = ((1 - alphas_cumprod[t]) ** 0.5).view(-1, 1, 1, 1)
                                        noise_pred = latents * sigma_t.view(-1, 1, 1, 1) + noise_pred * alpha_t.view(-1, 1, 1, 1)
                                        
        
                                    
                                        latents = (latents - noise_pred * sigma_t) / alpha_t
                                    else:
                                        latents = model.pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                                    
                                print(latents.shape)
                                       
                                if opt.cd_spatial_concat: # reshape back
                                    latents = einops.rearrange(latents, " B C (A H) (m n W) -> (B A) C (m H) (n W)", B=data['cond'].shape[0], A=5, m=3, n=2)
                                    # latents = einops.rearrange(latents, " B C (A H) (m n W) -> (B A) C (m H) (n W)", A=5, m=3, n=2)
                               
                                for attr_latents, attr in zip(latents, ordered_attr_list):
                                    
                                    decoded_attributes, decoded_images = decode_single_latents(pipeline_0123, attr_latents[None], attr_to_encode=attr)
                                    # NOTE: decoded_attributes is already mapped to their original range, not [0,1] or [-1,1]
                                    decoded_3channel_attr_image_dict.update({attr:decoded_images}) # which is for visualization, in range [-1,1]
                                    decoded_attributes_dict.update({attr:decoded_attributes}) # splatter attributes in original range 
                        
                            
                        else:
                            for attr, latents in latents_all_attr_dict.items():
                                # st()
                                # # print(f"[latents] {attr} - requires_grad: {latents.requires_grad}, grad: {getattr(latents, 'grad', None)}")
                                # if latents.requires_grad and latents.grad is not None:
                                #     print(f"Gradient of {attr}: {latents.requires_grad} -- {latents.grad.norm().item()}")
                                do_diffusion = True
                                if do_diffusion and attr=="rgbs":
                                    # st()
                                    with torch.no_grad():
                                        pipeline = model.pipe # .unet.requires_grad_(False).eval()
                                        prompt_embeds, cak = pipeline.prepare_conditions(cond, guidance_scale=guidance_scale)
                                        print(f"cak: {cak['cond_lat'].shape}") # always 64x64, not affected by cond size
                                        pipeline.scheduler.set_timesteps(75, device='cuda:0')
                                        # if opt.one_step_diffusion is not None:
                                        #     pipeline.scheduler.set_timesteps(opt.one_step_diffusion, device='cuda:0')
                                            
                                        timesteps = pipeline.scheduler.timesteps
                                    
                                        latents  = torch.randn_like(latents, device='cuda:0', dtype=torch.float32)
                                        
                                        attr_i = ordered_attr_list.index(attr)
                                        print(f"{ordered_attr_list} - {attr_i} = {attr}")

                                        domain_embeddings = torch.eye(5).to(latents.device)
                                        domain_embeddings = domain_embeddings[attr_i:attr_i+1]
                                        domain_embeddings = torch.cat([
                                                torch.sin(domain_embeddings),
                                                torch.cos(domain_embeddings)
                                            ], dim=-1)
                                        
                                        
                                        # latents_init = latents.clone().detach()
                                        for _, t in enumerate(timesteps):
                                            print(f"enumerate(timesteps) t={t}")
                                            # st()
                                            latent_model_input = torch.cat([latents] * 2)
                                            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

                                            # predict the noise residual
                                            noise_pred = pipeline.unet(
                                                latent_model_input,
                                                t,
                                                encoder_hidden_states=prompt_embeds,
                                                cross_attention_kwargs=cak,
                                                return_dict=False,
                                                class_labels=domain_embeddings,
                                            )[0]

                                            # perform guidance
                                            if True:
                                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                                            # compute the previous noisy sample x_t -> x_t-1
        
                                            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                                           
                                    
                                # st()
                            
                                decoded_attributes, decoded_images = decode_single_latents(pipeline_0123, latents, attr_to_encode=attr)
                                # NOTE: decoded_attributes is already mapped to their original range, not [0,1] or [-1,1]
                                decoded_3channel_attr_image_dict.update({attr:decoded_images}) # which is for visualization, in range [-1,1]
                                decoded_attributes_dict.update({attr:decoded_attributes}) # splatter attributes in original range 
                        
                
                    # # debug
                    # # Check gradients of the unet parameters 
                    # print(f"check unet parameters")
                    # for name, param in pipeline_0123.vae.named_parameters():
                    #     if param.requires_grad and param.grad is not None:
                    #         print(f"Parameter {name}") #s, Gradient norm: {param.grad.norm().item()}")
                
                    # print(f"check other model parameters")
                    # for name, param in pipeline_0123.unet.named_parameters():
                    #     if param.requires_grad and param.grad is not None:
                    #         print(f"Parameter {name}, Gradient norm: {param.grad.norm().item()}")
                    # print("grad at iter = ", i)
                    # st() # passed: no grad on any params
                
                    # order the attributes back into splatter image
                    # TODO: group_scale=False. do some check on dim
                    splatters_to_render = get_splatter_images_from_decoded_dict(decoded_attributes_dict, lgm_model=lgm_model, data=data, group_scale=group_scale)
                    # print("splatters_to_render.requires_grad: ", splatters_to_render.requires_grad)
                    # st()

                    # Render the decoded images using the splatter model
                    gs_results = render_from_decoded_images(gs, splatters_to_render, data=data, bg_color=global_bg_color) # NOTE: can do color_augmentation here
                    
                    # Step 3: Calculate the loss
                    # The target images should be the ground truth images you are trying to approximate
                    save_gt_path = None if i > 0 else os.path.join(output_path, f'{name}/gt')
                    results = calculate_loss(model, gs_results, data, save_gt_path=save_gt_path)
                    loss = results['loss']
                    psnr = results['psnr']
                    
                    if accelerator.is_main_process:
                        writer.add_scalar('decoded/loss', loss.item(), i)
                        writer.add_scalar('decoded/psnr', psnr.item(), i)
                    


                    additional_reg_on_splatter = False
                    if additional_reg_on_splatter:
                        # force rgb to be rgb
                        st()
                        mv_rgb_images_gt = einops.rearrange(data["input"], "b (m n) c h w -> b c (m h) (n w)", m=3, n=2) * 2 - 1
                        # reg_rgb = F.mse_loss(decoded_3channel_attr_image_dict["rgb"], mv_rgb_images_gt)
                        reg_rgb = F.mse_loss(splatter_original_Channel_image_to_encode["rgb"], mv_rgb_images_gt)
                        print(splatter_original_Channel_image_to_encode["rgb"].min(), splatter_original_Channel_image_to_encode["rgb"].max())
                        print(mv_rgb_images_gt.min(), mv_rgb_images_gt.max())
                        st()
                        
                        # force alpha to be alpha
                        mv_alpha_images_gt = einops.rearrange(data["masks_input"], "b (m n) c h w -> b c (m h) (n w)", m=3, n=2) * 2 - 1
                        # reg_rgb = F.mse_loss(decoded_3channel_attr_image_dict["opacity"], mv_alpha_images_gt)
                        reg_alpha = F.mse_loss(splatter_original_Channel_image_to_encode["opacity"], mv_alpha_images_gt)
                        
                        print(splatter_original_Channel_image_to_encode["rgb"].min(), splatter_original_Channel_image_to_encode["rgb"].max())
                        print(mv_rgb_images_gt.min(), mv_rgb_images_gt.max())
                        st()
                                             
                        ## force the depth to be the original depth
                        
                        loss = loss + reg_rgb + reg_alpha
                    
                    
                    if i==0:
                        acceptable_loss_threshold = 0.1 * loss
                   
                    # Step 4: Backward pass
                    # loss.backward()  # Compute gradient of the loss w.r.t. latents
                    
                   
                    # # Save the current state of the parameters before the update
                    # param_state_before = {name: param.clone() for name, param in model.named_parameters() if param.requires_grad}

                    # for attr, image in splatter_original_Channel_image_to_encode.items():
                    #     if image.requires_grad: # and image.grad is not None:
                    #         print(f"_/[Gradient] of [splatter image] {attr}: {image.requires_grad} -- {image.grad.norm().item()}")
                    #     # print(f"{attr} - requires_grad: {image.requires_grad}, grad: {getattr(image, 'grad', None)}")

                    
                    if opt.rendering_loss_on_splatter_to_encode:
                        to_encode_attributes_dict_init = {}
                        for attr, image in splatter_3Channel_image_to_encode.items():
                            decoded_attributes, _ = decode_single_latents(pipeline_0123, None, attr_to_encode=attr, mv_image=image)
                            to_encode_attributes_dict_init.update({attr:decoded_attributes}) # splatter attributes in original range
                        
                        
                        splatters_to_render = get_splatter_images_from_decoded_dict(to_encode_attributes_dict_init, lgm_model=lgm_model, data=data, group_scale=group_scale)
                        print("splatters_to_render.requires_grad: ", splatters_to_render.requires_grad)
                        # Render the decoded images using the splatter model
                        gs_results_to_encode = render_from_decoded_images(gs, splatters_to_render, data=data, bg_color=global_bg_color) # NOTE: can do color_augmentation here
                        
                        to_encode_results = calculate_loss(model, gs_results_to_encode, data, save_gt_path=save_gt_path)
                        to_encode_loss = to_encode_results['loss']
                        to_encode_psnr = to_encode_results['psnr']
                        
                        if i % save_iters == 0:
                            # save rendering results
                            save_path = os.path.join(output_path, f'{name}/{i}_encoder_input')
                            save_gs_rendered_images(gs_results_to_encode, fpath=save_path)

                            # Write PSNR value to the shared log file
                            with open(psnr_log_file, 'a') as f:
                                f.write(f"{save_path} - PSNR: {to_encode_psnr.item()}\n")
                        
                        loss = to_encode_loss + loss * opt.loss_weights_decoded_splatter
                        
                        if accelerator.is_main_process:
                            writer.add_scalar('to_encode/loss', to_encode_loss.item(), i)
                            writer.add_scalar('to_encode/psnr', to_encode_psnr.item(), i)
                            writer.add_scalar('total/loss', loss.item(), i)
                            writer.add_scalar('decoded/loss_weight', opt.loss_weights_decoded_splatter, i)
                        
                        # to_encode_loss.backward()
                    
                    
                    # Step 5: Update the latents
                    # optimizer.step()  # Perform a single optimization step
                    # check whether each latents has gradients
                    # # Check the parameters after the update
                    # for name, param in model.named_parameters():
                    #     if param.requires_grad:
                    #         # Ensure there is a gradient to avoid NoneType errors
                    #         if param.grad is not None:
                    #             # Calculate the change in the parameters
                    #             change = (param - param_state_before[name]).abs().sum().item()
                    #             if change != 0.:
                    #                 print(f"Parameter '{name}': Change in value after optimizer step: {change}")
                    #                 st()
                
                    # Step 6: Print the loss or log it to observe convergence
                    print(f'Iteration {i}, Loss: {loss.item()}')
                   
                    if i % save_iters == 0:
                        # save splatter image 
                        print(f"output_path={output_path}, name={name}")
                        save_3channel_splatter_images(decoded_3channel_attr_image_dict, fpath=os.path.join(output_path, f'{name}/{i}'), range_min=-1)
                        # save rendering results
                        save_path = os.path.join(output_path, f'{name}/{i}')
                        save_gs_rendered_images(gs_results, fpath=save_path)
                        
                        # Write PSNR value to the shared log file
                        with open(psnr_log_file, 'a') as f:
                            f.write(f"{save_path} - PSNR: {psnr.item()}\n")
                        
                        # also save the images to be encoded
                        save_3channel_splatter_images(splatter_3Channel_image_to_encode, fpath=os.path.join(output_path, f'{name}/{i}'), range_min=-1, suffix="to_encode")
                        # clip the image to encode
                        
                    # peviously only clip at save_iters!
                    if opt.clip_image_to_encode:
                        # for key, value in splatter_original_Channel_image_to_encode.items():
                        #     print("cliping the image to encode to [-1,1]: ", value.min(), value.max())
                        #     splatter_original_Channel_image_to_encode.update({key: torch.clip(value, -1, 1)})
                            
                        for attr, tensor in splatter_original_Channel_image_to_encode.items():
                            clipped_tensor = torch.clip(tensor.detach(), -1, 1).clone()
                            # Now we need to make sure that the optimizer updates this new tensor in the next iteration
                            # So we assign it back to the dictionary and reset requires_grad
                            splatter_original_Channel_image_to_encode[attr] = clipped_tensor.requires_grad_()
                            # print(f"[clip_image_to_encode]: {clipped_tensor.min(), clipped_tensor.max()}")

                        # You now need to make sure that the optimizer will update these new tensors in the next iteration
                        # Replace old tensors with new ones in optimizer.param_groups
                        optimizer.param_groups[0]['params'] = list(splatter_original_Channel_image_to_encode.values())
                            # st()



                    # Check for convergence or a stopping condition
                    
                    ## v1: loss threshold
                    # if loss.item() < acceptable_loss_threshold:
                    #     print("Lower than acceptable_loss_threshold, success!")
                    #     save_3channel_splatter_images(decoded_3channel_attr_image_dict, fpath=os.path.join(output_path, f'{name}/{i}_success'), range_min=-1)
                    #     save_gs_rendered_images(gs_results, fpath=os.path.join(output_path, f'{name}/{i}_success'))
                    #     break

                    ## v2: early stopping
                    # current_loss = loss.item()

                    # # Update the best loss and reset patience counter if current loss is better
                    # if current_loss < best_loss - delta:
                    #     best_loss = current_loss
                    #     patience_counter = 0
                    #     print(f"New best loss: {best_loss}")
                    # else:
                    #     patience_counter += 1

                    current_psnr = psnr.item()

                    # Update the best loss and reset patience counter if current loss is better
                    if current_psnr > best_psnr + delta:
                        best_psnr = current_psnr
                        patience_counter = 0
                        print(f"New best psnr: {best_psnr}")
                    else:
                        patience_counter += 1


                    # If no improvement for a number of iterations specified by patience_limit, stop
                    if patience_counter >= patience_limit:
                        print("Early stopping triggered")

                        save_3channel_splatter_images(splatter_3Channel_image_to_encode, fpath=os.path.join(output_path, f'{name}/{i}_success'), range_min=-1, suffix="to_encode")
                        save_3channel_splatter_images(decoded_3channel_attr_image_dict, fpath=os.path.join(output_path, f'{name}/{i}_success'), range_min=-1)
                        save_gs_rendered_images(gs_results, fpath=os.path.join(output_path, f'{name}/{i}_success'))
                        # st()
                        break
                    
                    
                
                print("Continue for the next scene")

                continue

                run_previous = False
                if run_previous:
                    decoded_attr_image_dict = {}

                    for attr_to_encode, (start_i, end_i) in attr_map.items():

                       
                        splatter_attr = splatters_mv[:,start_i:end_i,...]
                        print(f"Attr {attr_to_encode}: min={splatter_attr.min()} max={splatter_attr.max()}")

                        # if attr_to_encode == "rotation":
                        # if True:
                        # if attr_to_encode not in ["rgbs", "opacity", "z-depth"]: # passed "xy-offset"
                        # if attr_to_encode not in ["xy-offset"]: # passed  , "z-depth"
                        # if attr_to_encode not in ["z-depth"]: # passed 
                        if attr_to_encode not in ["rotation"]: # passed 
                        # if  "scale" not in attr_to_encode: # passed
                            # print(f"Using attr : {attr_to_encode}")
                            decoded_attr_image_dict[attr_to_encode] =  splatter_attr # TODO: currently skip the encoding and decoding of rotation for simplicity
                            continue
                        else:
                            print(f"Diffusing attr : {attr_to_encode}")
                        
                        sp_min, sp_max = None, None

                        # process the channels
                        if end_i - start_i == 1:
                            print(f"repeat attr {attr_to_encode} for 3 times")
                            splatter_attr = splatter_attr.repeat(1, 3, 1, 1) # [0,1]
                        elif end_i - start_i == 3:
                            pass
                        elif attr_to_encode == "xy-offset":
                            # ## normalize to [0,1]
                            # sp_min, sp_max =  -1., 1.
                            # splatter_attr = (splatter_attr - sp_min) / (sp_max - sp_min)
                            ## cat one more dim
                            splatter_attr = torch.cat((splatter_attr, 0.5 * torch.ones_like(splatter_attr[:,0:1,...])), dim=1)
                        elif attr_to_encode == "rotation":
                            # st() # assert 4 is on the last dim
                            # quaternion to axis angle
                            quat = einops.rearrange(splatter_attr, 'b c h w -> b h w c')
                            axis_angle = quaternion_to_axis_angle(quat)
                            splatter_attr = einops.rearrange(axis_angle, 'b h w c -> b c h w')
                            # st()

                        else:
                            raise ValueError(f"The dimension of {attr_to_encode} is problematic to encode")
                        
                        if "scale" in attr_to_encode:
                            # use log scale
                            splatter_attr = torch.log(splatter_attr)
                            
                            print(f"{attr_to_encode} log min={splatter_attr.min()} max={splatter_attr.max()}")
                            sp_min, sp_max =  -10., -2.
                            splatter_attr = (splatter_attr - sp_min) / (sp_max - sp_min) # [0,1]
                            splatter_attr = splatter_attr.clip(0,1)

                        elif attr_to_encode in ["z-depth", "xy-offset", "pos"] :
                            # sp_min, sp_max =  splatter_attr.min(), splatter_attr.max()
                            # sp_min, sp_max =  -1., 1.
                            sp_min, sp_max =  -0.7, 0.7
                            splatter_attr = (splatter_attr - sp_min) / (sp_max - sp_min)
                            # splatter_attr = splatter_attr.clip(0,1) 
                        
                       
                        print(f"Normed attr {attr_to_encode}: min={splatter_attr.min()} max={splatter_attr.max()}")
                        

                        sp_image = splatter_attr * 2 - 1 # [map to range [-1,1]]
                        print(f"Normed attr [-1, 1] {attr_to_encode}: min={sp_image.min()} max={sp_image.max()}")
                        
                        # Save image before encoding
                        mv_image = einops.rearrange((sp_image[0].clip(-1,1)+1).cpu().numpy()*127.5, 'c h w-> h w c').astype(np.uint8) 
                        Image.fromarray(mv_image).save(os.path.join(output_path, f'{name}/{attr_to_encode}_to_encode.png'))
                        
                        # encode: splatter attr -> latent 
                        # sp_image_original = sp_image.clone()
                        sp_image = scale_image(sp_image)
                        sp_image = pipeline.vae.encode(sp_image).latent_dist.sample() * pipeline.vae.config.scaling_factor
                        latents = scale_latents(sp_image)

                        #  decode: latents -> platter attr
                        latents1 = unscale_latents(latents)
                        image = pipeline_0123.vae.decode(latents1 / pipeline.vae.config.scaling_factor, return_dict=False)[0]
                        image = unscale_image(image)


                        # save decoded image
                        mv_image_numpy = einops.rearrange((image[0].clip(-1,1)+1).cpu().numpy()*127.5, 'c h w-> h w c').astype(np.uint8) 
                        Image.fromarray(mv_image_numpy).save(os.path.join(output_path, f'{name}/{attr_to_encode}_pred.png'))

                        # scale back to original range
                        mv_image = image # [b c h w], in [-1,1]
                        
                        # if attr_to_encode in[ "z-depth", "xy-offset"]:
                        #     sp_image_o = mv_image
                        #     # st() # NOTE: try clip z-depth? No need, they are already within the range
                        # else:
                        sp_image_o = 0.5 * (mv_image + 1) # [map to range [0,1]]

                        print(f"Decoded attr [-1,1] {attr_to_encode}: min={mv_image.min()} max={mv_image.max()}")
                        print(f"Decoded attr [0,1] {attr_to_encode}: min={sp_image_o.min()} max={sp_image_o.max()}")

                        if "scale" in attr_to_encode:
                            # v2
                            sp_image_o = sp_image_o.clip(0,1) 
                            sp_image_o = sp_image_o * (sp_max - sp_min) + sp_min
                            
                            print(f"Decoded attr not clip [0,1] {attr_to_encode}: min={sp_image_o.min()} max={sp_image_o.max()}")
                            sp_image_o = torch.exp(sp_image_o)
                            print(sp_min, sp_max)
                            print(f"Decoded attr [unscaled] {attr_to_encode}: min={sp_image_o.min()} max={sp_image_o.max()}")

                        # elif attr_to_encode == "z-depth":
                            # sp_image_o = sp_image_o * (sp_max - sp_min) + sp_min
                        
                        elif attr_to_encode in[ "z-depth", "xy-offset", "pos"]:
                            sp_image_o = sp_image_o * (sp_max - sp_min) + sp_min

                        if attr_to_encode == "xy-offset": 
                            sp_image_o = sp_image_o[:,:2] # FIXME: ...,2??
                        
                        if attr_to_encode == "rotation": 
                          
                            ag = einops.rearrange(sp_image_o, 'b c h w -> b h w c')
                            quaternion = axis_angle_to_quaternion(ag)
                            sp_image_o = einops.rearrange(quaternion, 'b h w c -> b c h w')
                            # st()
                        
                        if end_i - start_i == 1:
                            # print(torch.allclose(torch.mean(sp_image_o, dim=1, keepdim=True), sp_image_o))
                            # st()
                            print(f"Decoded attr [unscaled, before mean] {attr_to_encode}: min={sp_image_o.min()} max={sp_image_o.max()}")
                            sp_image_o = torch.mean(sp_image_o, dim=1, keepdim=True) # avg.
                        
                            # sp_image_o = torch.median(sp_image_o, dim=1, keepdim=True).values # 
                            # sp_image_o = torch.max(sp_image_o, dim=1, keepdim=True).values # .
                            # st()
                            print(f"Decoded attr [unscaled, after median] {attr_to_encode}: min={sp_image_o.min()} max={sp_image_o.max()}")
                        
                        # save in the dict
                        decoded_attr_image_dict.update({attr_to_encode:sp_image_o})


                        print(f"Decoded attr [unscaled] {attr_to_encode}: min={sp_image_o.min()} max={sp_image_o.max()}")
                        # st()
                    # save gt 6 input views
                    gt_white_images = einops.rearrange(data['input'], 'b (h2 w2) c h w -> b c (h2 h) (w2 w)', h2=3, w2=2) 
                    gt_image = einops.rearrange((gt_white_images[0].clip(-1,1)).cpu().numpy()*255, 'c h w-> h w c').astype(np.uint8) 
                    Image.fromarray(gt_image).save(os.path.join(output_path, f'{name}/gt.png'))

                    ## render splatter 
                    render_splatter_images = True
                    if not render_splatter_images:
                        continue

                    # # reshape to original splatter image shape for splatter rendering
                    # ## cat all attrs
                    # if not group_scale:
                    #     ordered_attr_list = ["xy-offset", "z-depth", # 0-3
                    #                         'opacity', # 3-4
                    #                         'scale-x', 'scale-y', 'scale-z', # 4-7
                    #                         "rotation", # 7-11
                    #                         "rgbs", # 11-14
                    #                         ] # must be an ordered list according to the channels
                    # else:
                    #     ordered_attr_list = ["pos", # 0-3
                    #                         'opacity', # 3-4
                    #                         'scale', # 4-7
                    #                         "rotation", # 7-11
                    #                         "rgbs", # 11-14
                    #                         ] # must be an ordered list according to the channels
                    attr_image_list = [decoded_attr_image_dict[attr] for attr in ordered_attr_list ]
                    # [print(t.shape) for t in attr_image_list]
                    splatter_mv = torch.cat(attr_image_list, dim=1)

                    ## reshape 
                    splatters_to_render = einops.rearrange(splatter_mv, 'b c (h2 h) (w2 w) -> b (h2 w2) c h w', h2=3, w2=2) 

                    
                    # gs.render:
                    ## decoded image
                    gaussians = fuse_splatters(splatters_to_render)
                    bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device) * 0.5
                    gs_results = model.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
                    
                    # save gs.rendered images
                    # st()
                    pred_images = gs_results['image'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    kiui.write_image(os.path.join(output_path, f'{name}/gs_render_rgb.png'), pred_images)
        

                    pred_alphas = gs_results['alpha'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                    pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                    kiui.write_image(os.path.join(output_path, f'{name}/gs_render_alpha.png'), pred_alphas)

                    ## gt splatter image
                    gaussians = fuse_splatters(data["splatters_output"])
                    bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device) * 0.5
                    gs_results = model.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
                    
                    # save gs.rendered images
                    # st()
                    pred_images = gs_results['image'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    kiui.write_image(os.path.join(output_path, f'{name}/gs_render_rgb_gt.png'), pred_images)
        

                    pred_alphas = gs_results['alpha'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                    pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                    kiui.write_image(os.path.join(output_path, f'{name}/gs_render_alpha_gt.png'), pred_alphas)


                    # skip the remaining inference: trained decoder from codes
                    continue
                    


            elif opt.codes_from_diffusion:
                
                print("---------- Begin original inference ---------------")
                        
                path =f'/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/{scene_name}/000.png'
            
                name = path.split('/')[-2]
               
                inference_on_unseen = True
                if inference_on_unseen: 
                    img = to_rgb_image(Image.open(requests.get("https://d.skis.ltd/nrp/sample-data/lysol.png", stream=True).raw))
                    name = "lysol"
                else:
                    img = to_rgb_image(Image.open(path)) 
                
                
                name = f"{i}_{name}"
                os.makedirs(os.path.join(output_path, name), exist_ok=True)

                img.save(os.path.join(output_path, f'{name}/cond.png'))
                cond = [img]
                print(img)
                
                prompt_embeds, cak = pipeline.prepare_conditions(cond, guidance_scale=4.0)
                print(f"cak: {cak['cond_lat'].shape}") # always 64x64, not affected by cond size
                pipeline.scheduler.set_timesteps(75, device='cuda:0')
                # if opt.one_step_diffusion is not None:
                #     pipeline.scheduler.set_timesteps(opt.one_step_diffusion, device='cuda:0')
                    
                timesteps = pipeline.scheduler.timesteps
            
                latents  = torch.randn([1, pipeline.unet.config.in_channels, 120, 80], device='cuda:0', dtype=torch.float32)
                latents_init = latents.clone().detach()

                with torch.no_grad():
                    # if opt.one_step_diffusion is not None:
                       
                    #     text_embeddings = prompt_embeds
                    #     t = torch.tensor([opt.one_step_diffusion]).to(timesteps.device)
                    #     print("ONE STEP DIFFUSION timestep =",t)
                    #     # st()
                    #     x = model.predict_x0(
                    #         latents, text_embeddings, t=t, guidance_scale=guidance_scale, 
                    #         cross_attention_kwargs=cak, scheduler=pipeline.scheduler, model='zero123plus')
                    #     latents = x
                        
                    #     timesteps = [] # skip the step-by-step inference

                    
                    lipschitz_analysis_zero123p = False

                    if opt.lipschitz_mode is not None and lipschitz_analysis_zero123p:
                        # get gt latents from encoder
                        # make input 6 views into a 3x2 grid
                        images = einops.rearrange(data['input'], 'b (h2 w2) c h w -> b c (h2 h) (w2 w)', h2=3, w2=2) 
                        latents = model.encode_image(images) # [B, self.pipe.unet.config.in_channels, 120, 80]

                        # add noise 
                        if opt.lipschitz_mode == "gaussian_noise":
                            noise = torch.randn_like(latents, device=latents.device)
                        elif opt.lipschitz_mode == "constant":
                            print(f"Adding constant lipschitz noise of scale {opt.lipschitz_coefficient}")
                            noise = torch.ones_like(latents, device=latents.device)
                        else:
                            raise ValueError ("invalid mode type for lipschitz analysis")

                        codes_gt = latents.clone()
                        latents += noise * opt.lipschitz_coefficient
                        latent_loss = F.mse_loss(latents, codes_gt)
                        print(f"latent loss = {latent_loss}") 

                        timesteps = [] # skip the step-by-step inference
                    
                    encode_splatter = False
                    if encode_splatter:

                        # reshape splatter 
                         # make input 6 views into a 3x2 grid
                        splatters_mv = einops.rearrange(data["splatters_output"], 'b (h2 w2) c h w -> b c (h2 h) (w2 w)', h2=3, w2=2) 
                        # vae.encode

                        ## RGB
                        # splatter_rgb = splatters_mv[:,11:14,...]
                        # image = splatter_rgb * 2 - 1 # [map to range [-1,1]]
                       

                        # # Opacity
                        # splatter_opacity = splatters_mv[:,3:4,...].repeat(1, 3, 1, 1) # [0,1]
                        # image = splatter_opacity * 2 - 1 # [map to range [-1,1]]

                        # gt_attr_keys = ['pos', 'opacity', 'scale', 'rotation', 'rgbs']
                        # start_indices = [0, 3, 4, 7, 11]
                        # end_indices = [3, 4, 7, 11, 14]
                        
                        attr_to_encode = "z-depth"
                        start_i, end_i = attr_map[attr_to_encode]          

                        splatter_attr = splatters_mv[:,start_i:end_i,...]
                        if end_i - start_i == 1:
                            splatter_attr = splatter_attr.repeat(1, 3, 1, 1) # [0,1]
                        elif end_i - start_i == 3:
                            pass
                        elif attr_to_encode == "xy-offset":
                            # st()
                            ## normalize to [0,1]
                            splatter_attr = (splatter_attr - splatter_attr.min()) / (splatter_attr.max() - splatter_attr.min())
                            ## cat one more dim
                            splatter_attr = torch.cat((splatter_attr, 0.5 * torch.ones_like(splatter_attr[:,0:1,...])), dim=1)
                        else:
                            raise ValueError(f"The dimension of {attr_to_encode} is problematic to encode")
                        
                        print(f"Attr {attr_to_encode}: min={splatter_attr.min()} max={splatter_attr.max()}")
                        if "scale" in attr_to_encode:
                            splatter_attr *= 20
                            splatter_attr = splatter_attr.clip(0,1)
                            print(f"New range of {attr_to_encode}: min={splatter_attr.min()} max={splatter_attr.max()}")
                        elif attr_to_encode == "z-depth":
                            splatter_attr = (splatter_attr - splatter_attr.min()) / (splatter_attr.max() - splatter_attr.min())
                            # splatter_attr = splatter_attr.clip(0,1)
                           
                        # st()

                        sp_image = splatter_attr * 2 - 1 # [map to range [-1,1]]
                        
                        # Save image
                        mv_image = einops.rearrange((sp_image[0].clip(-1,1)+1).cpu().numpy()*127.5, 'c h w-> h w c').astype(np.uint8) 
                        Image.fromarray(mv_image).save(os.path.join(output_path, f'{name}/{attr_to_encode}_to_encode.png'))
                        
                        # encode
                        sp_image = scale_image(sp_image)
                        sp_image = pipeline.vae.encode(sp_image).latent_dist.sample() * pipeline.vae.config.scaling_factor
                        latents = scale_latents(sp_image)

                        timesteps = [] # skip the step-by-step inference
                    
                    for i, t in enumerate(timesteps):
                        print(f"enumerate(timesteps) t={t}")
                        # st()
                        latent_model_input = torch.cat([latents] * 2)
                        latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

                        # predict the noise residual
                        noise_pred = pipeline.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=cak,
                            return_dict=False,
                        )[0]

                        # perform guidance
                        if True:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                    

                    # --------- 
                    print("Codes from diffusion (moved)")
                    data['codes'] = latents # torch.Size([1, 4, 120, 80])
                    print(f"code-diffusion: max={latents.max()} min={latents.min()} mean={latents.mean()}")
            
                    #####  # check latents
                    latents1 = unscale_latents(latents)
                   
                    # ### lgm deocder 
                    # z = latents1 / model.decoder.vae.config.scaling_factor
                    
                    # ud = model.decoder
                    # sample = ud.vae.post_quant_conv(z)
                    # latent_embeds = None
                    # sample = ud.decoder.conv_in(sample)
                    # upscale_dtype = next(iter(ud.decoder.up_blocks.parameters())).dtype
                    # sample = ud.decoder.mid_block(sample, latent_embeds)
                    # sample = sample.to(upscale_dtype)
                    # # up
                    # for i, up_block in enumerate(ud.decoder.up_blocks):
                    #     # print(f"{i}th upblock input: {sample.shape}")
                    #     sample = up_block(sample, latent_embeds)
                    
                    # # print(f"{i}th upblock output: {sample.shape}")
                    # # st()
                    
                    # sample = ud.decoder.conv_norm_out(sample) 
                    # sample = ud.decoder.conv_act(sample)
                    # image = ud.decoder.conv_out(sample)
                    
                    # ### --- [end] ---
                    
                    image = pipeline_0123.vae.decode(latents1 / pipeline.vae.config.scaling_factor, return_dict=False)[0]
                    image = unscale_image(image)

                    latents_init1 = unscale_latents(latents_init)
                    image_init = pipeline.vae.decode(latents_init1 / pipeline.vae.config.scaling_factor, return_dict=False)[0]
                    image_init = unscale_image(image_init)

                save_single_frames = False
                
                if save_single_frames:
                    mv_image = einops.rearrange((image[0].clip(-1,1)+1).cpu().numpy()*127.5, 'c (h2 h) (w2 w)-> (h2 w2) h w c', h2=3, w2=2).astype(np.uint8) 
                    for i, image in enumerate(mv_image):
                        image = rembg.remove(image).astype(np.float32) / 255.0
                        if image.shape[-1] == 4:
                            image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
                        Image.fromarray((image * 255).astype(np.uint8)).save(os.path.join(output_path, f'{name}/{i:03d}.png'))
                else:
                    mv_image = einops.rearrange((image[0].clip(-1,1)+1).cpu().numpy()*127.5, 'c h w-> h w c').astype(np.uint8) 
                    image = mv_image

                    white_bg = False
                    if white_bg:
                        image = rembg.remove(image).astype(np.float32) / 255.0
                
                        if image.shape[-1] == 4:
                            alpha_image = np.repeat((1 - image[..., 3:4]), repeats=3, axis=-1) # .astype(np.uint8).astype(np.float32)
                            # image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
                            image = image[..., :3] *(1 - alpha_image) + alpha_image
                    
                            Image.fromarray((alpha_image * 255).astype(np.uint8)).save(os.path.join(output_path, f'{name}/alpha.png'))
                        Image.fromarray((image * 255).astype(np.uint8)).save(os.path.join(output_path, f'{name}/pred.png'))

                        gt_white_images = einops.rearrange(data['input_white'], 'b (h2 w2) c h w -> b c (h2 h) (w2 w)', h2=3, w2=2) 
                        gt_image = einops.rearrange((gt_white_images[0].clip(-1,1)).cpu().numpy()*255, 'c h w-> h w c').astype(np.uint8) 
                        Image.fromarray(gt_image).save(os.path.join(output_path, f'{name}/gt_white.png'))
                    else:
                        Image.fromarray(image).save(os.path.join(output_path, f'{name}/pred.png'))

                        gt_white_images = einops.rearrange(data['input'], 'b (h2 w2) c h w -> b c (h2 h) (w2 w)', h2=3, w2=2) 
                        gt_image = einops.rearrange((gt_white_images[0].clip(-1,1)).cpu().numpy()*255, 'c h w-> h w c').astype(np.uint8) 
                        Image.fromarray(gt_image).save(os.path.join(output_path, f'{name}/gt.png'))


                print("---------- the above is original inference ---------------")
                
                # continue
                
                # debug_latent = True
                # if debug_latent:
                #     # Load the pipeline #TODO: this is only debug
                #     pipeline = DiffusionPipeline.from_pretrained(
                #         "sudo-ai/zero123plus-v1.1", custom_pipeline="/mnt/kostas-graid/sw/envs/chenwang/workspace/diffgan/training/modules/zero123plus.py",
                #         torch_dtype=torch.float32
                #     )
                #     pipeline.to('cuda:0')
                    
                # else:
                #     pipeline = model.pipe
                
                # pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                #         pipeline.scheduler.config, timestep_spacing='trailing'
                #     )
                # # TODO: try the default scheduler
                # pipeline.prepare()
                
                # guidance_scale = 4.0
                # cond_path=f'/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/{scene_name}/000.png'
                # img = to_rgb_image(Image.open(cond_path))
                    
                # img.save(f'{directory}/cond.jpg')
                # cond = [img]

                # prompt_embeds, cak = pipeline.prepare_conditions(cond, guidance_scale=4.0)
                # print(f"cak: {cak['cond_lat'].shape}") # always 64x64, not affected by cond size
                # model.pipe.scheduler.set_timesteps(75, device="cuda:0")
                # timesteps = model.pipe.scheduler.timesteps.to(torch.int64)
                # #TODO: check the timesteps every iter
                # print(timesteps)
                # # st()

                # latents = torch.randn([1, model.pipe.unet.config.in_channels, 120, 80], device="cuda:0", dtype=torch.float32)
                # latents_init = latents.clone().detach()
            
                # with torch.no_grad():
                    
                #     for _, t in enumerate(timesteps):
                #         latent_model_input = torch.cat([latents] * 2)
                #         latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

                #         # predict the noise residual
                #         noise_pred = pipeline.unet(
                #             latent_model_input,
                #             t,
                #             encoder_hidden_states=prompt_embeds,
                #             cross_attention_kwargs=cak,
                #             return_dict=False,
                #         )[0]

                #         # perform guidance
                #         if True:
                #             noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                #             noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                #         # noise_pred = predict_noise0_diffuser(pipeline.unet, latents, prompt_embeds, t, guidance_scale, cak, pipeline.scheduler)

                #         # compute the previous noisy sample x_t -> x_t-1
                #         latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0] # torch.Size([1, 4, 120, 80])
                    
                #     print("Codes from diffusion")
                #     data['codes'] = latents # torch.Size([1, 4, 120, 80])
                    
                #     if debug_latent:
                #         #####  # check latents
                #         latents1 = unscale_latents(latents)
                #         image = pipeline.vae.decode(latents1 / pipeline.vae.config.scaling_factor, return_dict=False)[0]
                #         image = unscale_image(image)
                        
                #         mv_image = einops.rearrange((image[0].clip(-1,1)+1).cpu().numpy()*127.5, 'c (h2 h) (w2 w)-> (h2 w2) h w c', h2=3, w2=2).astype(np.uint8) 
                #         for j, image in enumerate(mv_image):
                #             image = rembg.remove(image).astype(np.float32) / 255.0
                #             if image.shape[-1] == 4:
                #                 image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
                            
                #             Image.fromarray((image * 255).astype(np.uint8)).save(f'{directory}/{j:03d}.png')
        
            # elif not opt.codes_from_encoder:
            elif opt.codes_from_cache: # NOTE: make this more explicit
                ## ---- load or init code here ----
                
                if num_gpus==1:
                    codes = model.load_scenes(code_cache_dir, data, eval_mode=True)
                else:
                    codes = model.module.load_scenes(code_cache_dir, data, eval_mode=True)
                
                assert not (opt.one_step_diffusion is not None) and (opt.lipschitz_mode is not None)

                if opt.lipschitz_mode is not None:
                    if opt.lipschitz_mode == "gaussian_noise":
                        noise = torch.randn_like(codes, device=codes.device)
                    elif opt.lipschitz_mode == "constant":
                        print(f"Adding constant lipschitz noise of scale {opt.lipschitz_coefficient}")
                        noise = torch.ones_like(codes, device=codes.device)
                    else:
                        raise ValueError ("invalid mode type for lipschitz analysis")

                    # num_levels = 100
                    # disturb_level = torch.linspace(num_levels)/num_levels
                    codes_gt = codes.clone()
                    codes += noise * opt.lipschitz_coefficient
                    latent_loss = F.mse_loss(codes, codes_gt)
                    print(f"latent loss = {latent_loss}")
                    
                    

                if opt.one_step_diffusion is not None: 
                    t = torch.tensor([opt.one_step_diffusion]).to(codes.device)
                    
                    noise = torch.randn_like(codes, device=codes.device)
                    noisy_latents = model.pipe.scheduler.add_noise(codes, noise, t)

                    # get cond embed
                    path =f'/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/{scene_name}/000.png'
                    output_path = f"{opt.workspace}/zero123plus/codes_from_cache"
                    name = path.split('/')[-2]
                    os.makedirs(os.path.join(output_path, name), exist_ok=True)
                    # model.pipe.prepare()
                    guidance_scale = 4.0
                    img = to_rgb_image(Image.open(path))
                    img.save(os.path.join(output_path, f'{name}/cond.png'))
                    cond = [img]
                    print(img)
                    text_embeddings, cak = model.pipe.prepare_conditions(cond, guidance_scale=4.0)
                    # -------
           
                    print("ONE STEP DIFFUSION timestep =",t)
                    # st()
                    codes = model.predict_x0(
                        noisy_latents, text_embeddings, t=t, guidance_scale=guidance_scale, 
                        cross_attention_kwargs=cak, scheduler=model.pipe.scheduler, model='zero123plus')
                   
                
                
                data['codes'] = codes # torch.Size([1, 4, 120, 80])
                # st()
                
                print(f"code-optimized: max={codes.max()} min={codes.min()} mean={codes.mean()}")
                
                # ---- finish code init ----
            elif opt.codes_from_encoder:
                print("codes_from_encoder, are you sure?")
                # codes = model.encode_image(data['input'])
                # print(f"code-encoder: max={codes.max()} min={codes.min()} mean={codes.mean()}")

            else:
                raise ValueError("Not a valid source of latent")
            

            out = model(data)

            psnr = out['psnr']
            total_psnr += psnr.detach()
            loss = out['loss']
            total_loss += loss.detach()
            if 'loss_splatter' in out.keys():
                total_loss_splatter += out['loss_splatter'].detach()
            if 'loss_rendering' in out.keys():
                total_loss_rendering += out['loss_rendering'].detach()
            elif 'loss_alpha' in out.keys():
                total_loss_alpha += out["loss_alpha"].detach()
            if 'loss_lpips' in out.keys():
                total_loss_lpips += out['loss_lpips'].detach()
            
          
            # save some images
            # if accelerator.is_main_process:
            if True:

                gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                kiui.write_image(f'{directory}/image_gt.jpg', gt_images)

                pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                kiui.write_image(f'{directory}/image_pred.jpg', pred_images)

                pred_alphas = out['alphas_pred'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                kiui.write_image(f'{directory}/image_alpha.jpg', pred_alphas)
                
                ## save white images
                pred_images_white = pred_images * pred_alphas + 1 * (1 - pred_alphas)
                kiui.write_image(f'{directory}/image_pred_white.jpg', pred_images_white)

                gt_images_white = data['images_output_white'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                gt_images_white = gt_images_white.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images_white.shape[1] * gt_images_white.shape[3], 3) # [B*output_size, V*output_size, 3]
                kiui.write_image(f'{directory}/image_gt_white.jpg', gt_images_white)
                
        
                # # add write images for splatter to optimize
                # pred_images = out['images_opt'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                # pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                # kiui.write_image(f'{opt.workspace}/eval_ckpt/{i}_image_splatter_opt.jpg', pred_images)

                # pred_alphas = out['alphas_opt'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                # pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                # kiui.write_image(f'{opt.workspace}/eval_ckpt/{i}_image_splatter_opt_alpha.jpg', pred_alphas)

                if len(opt.plot_attribute_histgram) > 0:
                    for splatters_pred_key in ['splatters_from_code']:
                        if splatters_pred_key == 'splatters_from_code':
                            splatters = out[splatters_pred_key]
                        else:
                            raise NotImplementedError
                        
                        gaussians = fuse_splatters(splatters)
                        gt_gaussians = fuse_splatters(data['splatters_output'])
                        
                        color_pairs = [('pink', 'teal'), ("red", "green"), ("orange", "blue"), ('purple', 'yellow'), ('cyan', 'brown')]

                        attr_map = {key: (si, ei, color_pair) for key, si, ei, color_pair in zip (gt_attr_keys, start_indices, end_indices, color_pairs)}
        
                        for attr in opt.plot_attribute_histgram:
                        
                            start_i, end_i, (gt_color, pred_color) = attr_map[attr]
                            # if opt.verbose_main:
                            #     print(f"plot {attr} in dim ({start_i}, {end_i})")
                            
                            gt_attr_flatten =  gt_gaussians[..., start_i:end_i] # [B, L, C]
                            pred_attr_flatten = gaussians[..., start_i:end_i]
                            
                            if attr in ['scale', 'opacity']:
                                gt_attr_flatten = torch.log(gt_attr_flatten).permute(0,2,1) # [B, C, L]
                                pred_attr_flatten = torch.log(pred_attr_flatten).permute(0,2,1) 
                                gt_attr_flatten = gt_attr_flatten.flatten().detach().cpu().numpy()
                                pred_attr_flatten = pred_attr_flatten.flatten().detach().cpu().numpy()

                                
                            else:
                                ## cannot flatten due to their meaning
                                print(f"not support the plotting of __{attr}__ yet")
                                continue
                            
                            # Manually define bin edges
                            bin_edges = np.linspace(min(min(gt_attr_flatten), min(pred_attr_flatten)), max(max(gt_attr_flatten), max(pred_attr_flatten)), num=50)

                            plt.hist(gt_attr_flatten, bins=bin_edges, color=gt_color, alpha=0.7, label=f'{attr}_gt')
                            plt.hist(pred_attr_flatten, bins=bin_edges, color=pred_color, alpha=0.3, label=f'{attr}_pred')
                            
                            # Add labels and legend
                            plt.xlabel('Value')
                            plt.ylabel('Frequency')
                            plt.legend()

                            # Save the plot as an image file (e.g., PNG)
                            name = f'histogram_epoch_batch{i}_{splatters_pred_key}_{attr}'
                            if attr == "scale":
                                name += f"_{opt.scale_act}_bias{opt.scale_act_bias}"
                            
                            if opt.normalize_scale_using_gt:
                                name += "normed_on_gt"
                                
                            plt.title(f'{name}')
                            plt.savefig(f'{opt.workspace}/eval_ckpt/{i}_{name}.jpg')
                        
                            # Clear the figure
                            plt.clf()
                    
    
        torch.cuda.empty_cache()

        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        if accelerator.is_main_process:
            total_psnr /= len(test_dataloader)
            # accelerator.print(f"[eval] epoch: {epoch} psnr: {psnr:.4f}")
            total_loss /= len(test_dataloader)
            total_loss_splatter /= len(test_dataloader)
            total_loss_rendering /= len(test_dataloader)
            total_loss_alpha /= len(test_dataloader)
            total_loss_lpips /= len(test_dataloader)
            
            accelerator.print(f"[eval] ckpt loss: {total_loss.item():.6f} psnr: {total_psnr.item():.4f} splatter_loss: {total_loss_splatter:.4f} rendering_loss: {total_loss_rendering:.4f} alpha_loss: {total_loss_alpha:.4f} lpips_loss: {total_loss_lpips:.4f} ")
           

if __name__ == "__main__":
    
    ### Ignore the FutureWarning from pipeline_stable_diffusion.py
    warnings.filterwarnings("ignore", category=FutureWarning, module="pipeline_stable_diffusion")
    
    main()
    
    # Reset the warning filter to its default state (optional)
    warnings.resetwarnings()
