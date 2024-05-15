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

from ipdb import set_trace as st

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


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
    }
def load_splatter_png_as_original_channel_images_to_encode(splatter_dir, suffix="to_encode", device="cuda", ext="png"):
    # valid siffices: ["decoded", "to_encode"]
    # print(f"Loading {suffix}_{ext} files")
    # NOTE: since we are loading png not ply, no need to do deactivation
    splatter_3Channel_image = {}
    
    for attr in ordered_attr_list:
        # im_path = os.path.join(splatter_dir, f"{attr}_{suffix}.png")
        if not isinstance(splatter_dir, str):
            print("BUG")
            print(splatter_dir)
            raise TypeError(f"Expected splatter_dir to be a string, got {type(splatter_dir).__name__} instead")
            exit()

        # try:
        if True:
            # print(f"splatter_dir: {splatter_dir}")
            # print(f"{attr}_{suffix}.{ext}")
            im_path = os.path.join(splatter_dir, f"{attr}_{suffix}.{ext}")
        # except:
        #     print(f"splatter_dir: {splatter_dir}")
        #     print(f"{attr}_{suffix}.{ext}")
            # st()
        
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
        # print("images shape: ", image.shape) # (384, 256, 3)
        
        
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
        new_value = einops.rearrange(value, "h w c -> c h w") # [None]
        # print("new shape: ", new_value.shape)
        # print(new_value.min(), new_value.max())
        
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
        # splatter_3Channel_image.update({key: new_value.to(device)})
        splatter_3Channel_image.update({key: new_value})
        # print(new_value.shape)
    
    # depth_min, depth_max = 0, 3 
    # sp_min_max_dict["z-depth"] = depth_min, depth_max 
  
    assert set(ordered_attr_list) == set(splatter_3Channel_image.keys())
    
    return splatter_3Channel_image


class ObjaverseDataset(Dataset):

    def __init__(self, opt: Options, training=True, prepare_white_bg=False):
        
        self.opt = opt
        if self.opt.model_type == 'LGM':
            self.opt.bg = 1.0
        self.training = training
        self.prepare_white_bg = prepare_white_bg

      
        # # TODO: load the list of objects for training
        # self.items = ["/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/0a9b36d36e904aee8b51e978a7c0acfd"]
        # self.items_splatter_gt = ['/home/xuyimeng/Repo/LGM/data/splatter_gt_full/00000-hydrant-eval_pred_gs_1800_0']
        # # with open('TODO: file containing the list', 'r') as f:
        # #     for line in f.readlines():
        # #         self.items.append(line.strip())

        self.data_path_rendering = {}
        # self.data_path_splatter_gt = {}
        self.data_path_vae_splatter = {}
        
        # take the vae splatter "to_encode" ("decoded" is not required for now)
        
        # if not opt.data_path_splatter_gt.endswith('9000-9999'):
        #     scene_path_pattern = os.path.join(opt.data_path_splatter_gt, "*", "9000-9999", "*")
        # else:
        #     scene_path_pattern = os.path.join(opt.data_path_splatter_gt, "*")
        
        scene_path_pattern = os.path.join(opt.data_path_vae_splatter, "*", "zero123plus/outputs_v3_inference_my_decoder", "*")
        
        all_scene_paths = sorted(glob.glob(scene_path_pattern))
            
        for scene_path in all_scene_paths:

            if self.opt.overfit_one_scene and len(self.data_path_vae_splatter) > 3:
                break
   
            if not os.path.isdir(scene_path):
                continue
    
            scene_name = scene_path.split('/')[-1]
            if scene_name in self.data_path_vae_splatter.keys():
                continue
            
            final_optimized = return_final_scene(scene_path, acceptable_epoch=300, verbose=True)
            if final_optimized == None:
                print(f"scene {scene_name} is not optimized to the end point")
                print(scene_path)
                st()
                
                continue
            
            
            vae_splatter_folder = os.path.join(scene_path, final_optimized)

    
            if len(os.listdir(vae_splatter_folder)) == 22:
                
                rendering_folder = os.path.join(opt.data_path_rendering, scene_name.split("_")[-1])
                try:
                    assert len(os.listdir(rendering_folder)) >= 112   
                except:
                    print(f"{rendering_folder}: < 56 views of rendering")
                    continue
                
                if not isinstance(vae_splatter_folder, str):
                    print("BUG")
                    print(vae_splatter_folder)
                    st()
                    # raise TypeError(f"Expected splatter_dir to be a string, got {type(vae_splatter_folder).__name__} instead")
                    # exit()
                self.data_path_vae_splatter[scene_name] = vae_splatter_folder
                self.data_path_rendering[scene_name] = rendering_folder

            elif os.path.isdir(os.path.join(scene_path, "200")) and len(os.listdir(os.path.join(scene_path, "200"))) == 22:
                vae_splatter_folder = os.path.join(scene_path, "200")
                print("Can use the 200 epoch!")
                  
                rendering_folder = os.path.join(opt.data_path_rendering, scene_name.split("_")[-1])
                try:
                    assert len(os.listdir(rendering_folder)) >= 112   
                except:
                    print(f"{rendering_folder}: < 56 views of rendering")
                    continue
                
                
                self.data_path_vae_splatter[scene_name] = vae_splatter_folder
                if not isinstance(vae_splatter_folder, str):
                    print("BUG-200")
                    print(vae_splatter_folder)
                    st()
                    
                self.data_path_rendering[scene_name] = rendering_folder
                # print(vae_splatter_folder, len(os.listdir(vae_splatter_folder)) == 22)
                # st()

        assert len(self.data_path_vae_splatter) == len(self.data_path_rendering)
        
        # self.items = [k for k in self.data_path_splatter_gt.keys()]
        all_items = [k for k in self.data_path_vae_splatter.keys()]
        num_val = min(50, len(all_items)//2) # when using small dataset to debug
        if self.training:
            self.items = all_items # NOTE: all scenes are used for training and val
            if self.opt.overfit_one_scene:
                # print(f"[WARN]: always fetch the 0th item. For debug use only")
                # self.items = all_items[:1]
                print(f"[WARN]: always fetch the 1th item. For debug use only")
                self.items = all_items[1:2]
        else:
            self.items = all_items
            if self.opt.overfit_one_scene:
                # print(f"[WARN]: always fetch the 0th item. For debug use only")
                # self.items = all_items[:1]
                print(f"[WARN]: always fetch the 1th item. For debug use only")
                self.items = all_items[1:2]
        
        
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


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        
        scene_name = self.items[idx]
        if self.opt.overfit_one_scene:
            print(f"[WARN]: always fetch the {idx} item. For debug use only")
        
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

        # TODO: choose views, based on your rendering settings
        # if self.training:
        #     # input views are in (36, 72), other views are randomly selected
        #     vids = np.random.permutation(np.arange(36, 73))[:self.opt.num_input_views].tolist() + np.random.permutation(100).tolist()
        # else:
        #     # fixed views
        #     vids = np.arange(36, 73, 4).tolist() + np.arange(100).tolist()
        if self.training:
            vids = np.arange(1, 7)[:self.opt.num_input_views].tolist() + np.random.permutation(56).tolist()
            # vids = np.arange(1, 7)[:self.opt.num_input_views].tolist() + [1]*56
        else:
            vids = np.arange(1, 7)[:self.opt.num_input_views].tolist() + np.arange(7, 56).tolist()

        cond_path = os.path.join(uid, f'000.png')
        from PIL import Image
        cond = np.array(Image.open(cond_path).resize((self.opt.input_size, self.opt.input_size)))
        # print(f"cond size:{Image.open(cond_path)}")
        mask = cond[..., 3:4] / 255
        cond = cond[..., :3] * mask + (1 - mask) * int(self.opt.bg * 255)
        results['cond'] = cond.astype(np.uint8)

        # load splatter gt
        # splatter_uid = self.items_splatter_gt[idx]
       
        splatter_images_multi_views = []
        
        # for input_id in vids[:self.opt.num_input_views]:
        #     sf = os.path.join(splatter_uid, f"splatter_{input_id-1}.ply")
        #     if self.opt.verbose:
        #         print(f"sf:{sf}")
        #     splatter_im = load_ply(sf)
        #     splatter_images_multi_views.append(splatter_im)
        
        # splatter_images_mv = torch.stack(splatter_images_multi_views, dim=0) # # [6, 16384, 14]
        # splatter_res = int(math.sqrt(splatter_images_mv.shape[-2]))
        # # print(f"splatter_res: {splatter_res}")
        # ## when saving the splatter image in model_fix_pretrained.py: x = einops.rearrange(self.splatter_out, 'b v c h w -> b v (h w) c')
        # splatter_images_mv = einops.rearrange(splatter_images_mv, 'v (h w) c -> v c h w', h=splatter_res, w=splatter_res)
        # results['splatters_output'] = splatter_images_mv
        # # print(results['splatters_output'].shape) # [6, 14, 128, 128])
         
        # print("Load splatter png from ", splatter_uid)
        splatter_original_Channel_mvimage_dict = load_splatter_png_as_original_channel_images_to_encode(splatter_uid, suffix="to_encode", ext="png")
        results.update(splatter_original_Channel_mvimage_dict)
        
        print("vids:", vids)
        for vid in vids:

            image_path = os.path.join(uid, f'{vid:03d}.png')
            camera_path = os.path.join(uid, f'{vid:03d}.npy')

            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255 # [512, 512, 4] in [0, 1]
            # print("images shape: ", image.shape) # 320x320x4
            image = torch.from_numpy(image)

            cam = np.load(camera_path, allow_pickle=True).item()
            
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
            if self.training and (vid_cnt == self.opt.num_views):
                break

        if vid_cnt < self.opt.num_views:
            print(f"vid_cnt{vid_cnt} < self.opt.num_views{self.opt.num_views}")
            print(f'[WARN] dataset {uid}: not enough valid views, only {vid_cnt} views found!')
            n = self.opt.num_views - vid_cnt
            images = images + [images[-1]] * n
            if self.prepare_white_bg:
                images_white = images_white + [images_white[-1]] * n
            masks = masks + [masks[-1]] * n
            cam_poses = cam_poses + [cam_poses[-1]] * n
          
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
        if self.prepare_white_bg:
            images_input_white = F.interpolate(images_white[:self.opt.num_input_views].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        cam_poses_input = cam_poses[:self.opt.num_input_views].clone()

        # data augmentation
        if self.training:
            # apply random grid distortion to simulate 3D inconsistency
            if random.random() < self.opt.prob_grid_distortion:
                images_input[1:] = grid_distortion(images_input[1:])
            # apply camera jittering (only to input!)
            if random.random() < self.opt.prob_cam_jitter:
                cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:])

        # FIXME: we don't need this for zero123plus?
        if self.opt.model_type == 'LGM':
            images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # resize render ground-truth images, range still in [0, 1]
        render_input_views = self.opt.render_input_views
        
        results['images_output'] = F.interpolate(images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        
        if self.prepare_white_bg:
            results['images_output_white'] = F.interpolate(images_white, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        if self.opt.verbose:
            print(f"images_input:{images_input.shape}") # [20, 3, input_size, input_size] input_size=128
            print("images_output", results['images_output'].shape) # [20, 3, 512, 512]
        
        results['masks_output'] = F.interpolate(masks.unsqueeze(1), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]
        if not render_input_views:
            results['images_output'] = results['images_output'][self.opt.num_input_views:]
            results['masks_output'] = results['masks_output'][self.opt.num_input_views:]

        
        # print(f"images_output.shape:{results['images_output'].shape}")
            

        # build rays for input views
        if self.opt.model_type == 'LGM':
            rays_embeddings = []
            for i in range(self.opt.num_input_views):
                rays_o, rays_d = get_rays(cam_poses_input[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
                rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
                rays_embeddings.append(rays_plucker)

            rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
            final_input = torch.cat([images_input, rays_embeddings], dim=1) # [V=4, 9, H, W]
            results['input'] = final_input
        else:
            results['input'] = images_input
            results['masks_input'] = masks_input
            # results['input_white'] = images_input_white
            
            
            lgm_images_input = F.interpolate(images[:self.opt.num_input_views].clone(), size=(256, 256), mode='bilinear', align_corners=False) # [V, C, H, W]
            lgm_images_input = TF.normalize(lgm_images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

            ## for adding additonal input for lgm
            rays_embeddings = []
            for i in range(self.opt.num_input_views):
                rays_o, rays_d = get_rays(cam_poses_input[i], 256, 256, self.opt.fovy) # [h, w, 3]
                rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
                rays_embeddings.append(rays_plucker)

            rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
            final_input = torch.cat([lgm_images_input, rays_embeddings], dim=1) # [V=4, 9, H, W]
            results['input_lgm'] = final_input

        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction

        # should use the same world coord as gs renderer to convert depth into world_xyz
        results['c2w_colmap'] = cam_poses[:self.opt.num_input_views].clone() 

        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
        
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        if render_input_views:
            results['cam_view'] = cam_view
            results['cam_view_proj'] = cam_view_proj
            results['cam_pos'] = cam_pos

        else:
            results['cam_view'] = cam_view[self.opt.num_input_views:]
            results['cam_view_proj'] = cam_view_proj[self.opt.num_input_views:]
            results['cam_pos'] = cam_pos[self.opt.num_input_views:]

        results['scene_name'] = scene_name
        # print(f"returning scene_name:{scene_name}")


        return results