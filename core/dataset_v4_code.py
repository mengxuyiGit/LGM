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
        self.data_path_splatter_gt = {}
        
        # check the integrity of each splatter gt
        
        if not opt.data_path_splatter_gt.endswith('9000-9999'):
            scene_path_pattern = os.path.join(opt.data_path_splatter_gt, "*", "9000-9999", "*")
        else:
            scene_path_pattern = os.path.join(opt.data_path_splatter_gt, "*")
        
        all_scene_paths = sorted(glob.glob(scene_path_pattern))
            
        for scene_path in all_scene_paths:
            pattern = os.path.join(scene_path, 'eval_pred_gs_*_es')
            es_folder =  glob.glob(pattern)
            try:
                assert len(es_folder) >= 1
            except:
                if self.opt.verbose:
                    print(f"{scene_path} does not contain exactly one early stop ckpt")
                continue
                
            splatter_gt_folder = es_folder[0]

            scene_name = scene_path.split('/')[-1]
            
            if scene_name in self.data_path_splatter_gt.keys():
                continue
            
            if len(os.listdir(splatter_gt_folder)) == 7:
                
                rendering_folder = os.path.join(opt.data_path_rendering, scene_name)
                try:
                    assert len(os.listdir(rendering_folder)) >= 112   
                except:
                    print(f"{rendering_folder}: < 56 views of rendering")
                    continue
                
                self.data_path_splatter_gt[scene_name] = splatter_gt_folder
                self.data_path_rendering[scene_name] = rendering_folder

        assert len(self.data_path_splatter_gt) == len(self.data_path_rendering)
        
        # self.items = [k for k in self.data_path_splatter_gt.keys()]
        all_items = [k for k in self.data_path_splatter_gt.keys()]
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
        splatter_uid = self.data_path_splatter_gt[scene_name] 
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
        
        # spaltter_files = glob.glob(os.path.join(splatter_uid, "splatter_*.ply")) # TODO: make this expresion more precise: only load the ply files with numbers ## FIXME: 
        # for sf in spaltter_files:
        
        for input_id in vids[:self.opt.num_input_views]:
            sf = os.path.join(splatter_uid, f"splatter_{input_id-1}.ply")
            if self.opt.verbose:
                print(f"sf:{sf}")
            splatter_im = load_ply(sf)
            splatter_images_multi_views.append(splatter_im)
            # print(splatter_im.shape) # ([16384, 14])
        
        splatter_images_mv = torch.stack(splatter_images_multi_views, dim=0) # # [6, 16384, 14]
        splatter_res = int(math.sqrt(splatter_images_mv.shape[-2]))
        # print(f"splatter_res: {splatter_res}")
        ## when saving the splatter image in model_fix_pretrained.py: x = einops.rearrange(self.splatter_out, 'b v c h w -> b v (h w) c')
        splatter_images_mv = einops.rearrange(splatter_images_mv, 'v (h w) c -> v c h w', h=splatter_res, w=splatter_res)
        results['splatters_output'] = splatter_images_mv
        # print(results['splatters_output'].shape) # [6, 14, 128, 128])
        
        for vid in vids:

            image_path = os.path.join(uid, f'{vid:03d}.png')
            camera_path = os.path.join(uid, f'{vid:03d}.npy')

            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255 # [512, 512, 4] in [0, 1]
            image = torch.from_numpy(image)

            cam = np.load(camera_path, allow_pickle=True).item()
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
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]

        images_input = F.interpolate(images[:self.opt.num_input_views].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
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