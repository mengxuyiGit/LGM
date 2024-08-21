import json
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
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



def load_images_from_folder_debug(my_tarfile, name, load_normal=False):
    """
    Load all images in a folder into a list of numpy arrays.
    """
    images = []
    normals = []

    for i in range(40):
        if i==25 or i==26:
            continue
        
        img_path =f'{name}/{i:05d}/{i:05d}.png'
        normal_path =f'{name}/{i:05d}/{i:05d}_nd.exr'

        # print(img_path, os.path.exists(img_path), normal_path, os.path.exists(normal_path))

        try:
            # img_data = io.BytesIO(my_tarfile.extractfile(img_path).read())
            img = Image.open(img_path)
            images.append(np.array(img, dtype=np.uint8))
            # images.append(np.array(my_tarfile.extractfile(img_path).read(), dtype=np.uint8))
            # with Image.open(img_path) as img:
            #     # images.append(np.array(img.resize((size, size)), dtype=np.uint8))
            #     images.append(np.array(img, dtype=np.uint8))
            #     # print(images[-1].shape)

            if load_normal:
                
                # my_tarfile.extract(normal_path, path=f'temp')
                # normald = cv2.imread(f'temp/{normal_path}',-1)
                normald = cv2.imread(normal_path,-1)    
                # normald = np.array(my_tarfile.extractfile(normal_path).read())
                normal = normald[...,:3]
                normal_norm = (np.linalg.norm(normal, 2, axis=-1, keepdims= True))
                normal = normal / normal_norm
                normal = normal[...,[2,0,1]]
                normal[...,[0,1]] = -normal[...,[0,1]]
                normals.append(((normal+1)/2*255).astype('uint8'))
        except:
            return None, None
    return images, normals


def load_camera_poses_debug(my_tarfile, name):
    """
    Load camera poses from a file. Modify this function based on the format of your camera poses file.
    """
    # Example: Load camera poses from a JSON file
    # You need to replace this with the appropriate method to load your camera poses
    poses = []
    for i in range(40):
        if i==25 or i==26:
            continue
        try:
            pose = {}
            c2w = np.eye(4).astype('float32')
            file_path =f'{name}/{i:05d}/{i:05d}.json'
            # my_tarfile.extract(file_path, path=f'temp')
            
            
            # print("load_camera_poses_debug", file_path)
            with open(f'{file_path}', 'r') as file:
                temp = json.load(file)
            pose['fov'] = np.array([temp['x_fov'],temp['y_fov']],dtype=np.float32)
            c2w[:3,:3] = np.stack((temp['x'],temp['y'],temp['z']),axis=1)
            c2w[:3,3] = np.array(temp['origin'],dtype=np.float32)
            pose['bbox'] = np.array(temp['bbox'],dtype=np.float32)
            pose['c2w'] = c2w
            # print(c2w)
            poses.append(pose)
        except:
            return
    return poses


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
        normals = []
        depths = []
        
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
            
        numerical_value = 39
        
        ## fix input views
        fixed_input_views = np.arange(1,7).tolist()
        assert self.opt.num_input_views == 6
        if self.training:
            vids = fixed_input_views[:self.opt.num_input_views] + np.random.permutation(numerical_value+1).tolist()
        else:
            vids = fixed_input_views[:self.opt.num_input_views] + np.arange(numerical_value+1).tolist() # fixed order
            print(vids)
        
        final_vids = []
        
        
        # uid =  "/home/xuyimeng/Data/gobjverse/0/11274"
        for vid in vids:
    
            try:
                # tar_path = os.path.join(self.opt.data_path, f"{uid}.tar")
                # uid_last = uid.split('/')[1]

                # image_path = os.path.join(uid_last, 'campos_512_v4', f"{vid:05d}/{vid:05d}.png")
                # meta_path = os.path.join(uid_last, 'campos_512_v4', f"{vid:05d}/{vid:05d}.json")
                # # albedo_path = os.path.join(uid_last, 'campos_512_v4', f"{vid:05d}/{vid:05d}_albedo.png") # black bg...
                # # mr_path = os.path.join(uid_last, 'campos_512_v4', f"{vid:05d}/{vid:05d}_mr.png")
                # nd_path = os.path.join(uid_last, 'campos_512_v4', f"{vid:05d}/{vid:05d}_nd.exr")
                
                image_path =f'{uid}/{vid:05d}/{vid:05d}.png'
                nd_path =f'{uid}/{vid:05d}/{vid:05d}_nd.exr'
                meta_path = f'{uid}/{vid:05d}/{vid:05d}.json'
                
                # with tarfile.open(tar_path, 'r') as tar:
                #     with tar.extractfile(image_path) as f:
                #         image = np.frombuffer(f.read(), np.uint8)
                #     with tar.extractfile(meta_path) as f:
                #         meta = json.loads(f.read().decode())
                #     with tar.extractfile(nd_path) as f:
                #         nd = np.frombuffer(f.read(), np.uint8)
                            
                # image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255 # [512, 512, 4] in [0, 1]
                # image = torch.from_numpy(image) 
                with open(image_path, 'rb') as f:
                    image = np.frombuffer(f.read(), np.uint8)
                image = torch.from_numpy(cv2.imdecode(image, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255) # [512, 512, 4] in [0, 1]
                # print(image.shape)
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                # print(meta.keys())
                # # nd = cv2.imread(nd_path,-1)  
                # import imageio
                # nd = imageio.imread(nd_path)  
                # print(nd.shape)
                with open(nd_path, "rb") as f:
                    nd = np.frombuffer(f.read(), np.uint8)
            


                c2w = np.eye(4)
                c2w[:3, 0] = np.array(meta['x'])
                c2w[:3, 1] = np.array(meta['y'])
                c2w[:3, 2] = np.array(meta['z'])
                c2w[:3, 3] = np.array(meta['origin'])
                c2w = torch.tensor(c2w, dtype=torch.float32).reshape(4, 4) # until here, same as Lara 
                
                nd = cv2.imdecode(nd, cv2.IMREAD_UNCHANGED).astype(np.float32) # [512, 512, 4] in [-1, 1]
                normal = nd[..., :3] # in [-1, 1], bg is [0, 0, 1]
                depth = nd[..., 3] # in [0, +?), bg is 0

                # rectify normal directions
                normal = normal[..., ::-1]
                normal[..., 0] *= -1
                normal = torch.from_numpy(normal.astype(np.float32)).nan_to_num_(0) # there are nans in gt normal... 
                depth = torch.from_numpy(depth.astype(np.float32)).nan_to_num_(0)
                
            except Exception as e:
                print(f'[WARN] dataset {uid} {vid}: {e}')
                continue
            
            # blender world + opencv cam --> opengl world & cam
            c2w[1] *= -1
            c2w[[1, 2]] = c2w[[2, 1]]
            c2w[:3, 1:3] *= -1 # invert up and forward direction

            image = image.permute(2, 0, 1) # [4, 512, 512]
            mask = image[3:4] # [1, 512, 512]

            image = image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg

            image = image[[2,1,0]].contiguous() # bgr to rgb

            normal = normal.permute(2, 0, 1) # [3, 512, 512]
            normal = normal * mask # to [0, 0, 0] bg

            images.append(image)
            normals.append(normal)
            depths.append(depth)
            masks.append(mask.squeeze(0))
            cam_poses.append(c2w)

            vid_cnt += 1
            if vid_cnt == self.opt.num_views:
                break

        if vid_cnt < self.opt.num_views:
            print(f'[WARN] dataset {uid}: not enough valid views, only {vid_cnt} views found!')
            n = self.opt.num_views - vid_cnt
            images = images + [images[-1]] * n
            normals = normals + [normals[-1]] * n
            depths = depths + [depths[-1]] * n
            masks = masks + [masks[-1]] * n
            cam_poses = cam_poses + [cam_poses[-1]] * n
          
        images = torch.stack(images, dim=0) # [V, 3, H, W]
        normals = torch.stack(normals, dim=0) # [V, 3, H, W]
        depths = torch.stack(depths, dim=0) # [V, H, W]
        masks = torch.stack(masks, dim=0) # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]

        # normalized camera feats as in paper (transform the first pose to a fixed position)
        radius = torch.norm(cam_poses[0, :3, 3])
        cam_poses[:, :3, 3] *= self.opt.cam_radius / radius
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]
        
        
        # print("normals range: ", normals.permute(0,2,3,1).reshape(-1,3).max(dim=0)[0], normals.permute(0,2,3,1).reshape(-1,3).min(dim=0)[0]) # in range (-1,1)
        # TODO: normalize normal to [0, 1], AS InstantMesh does
        # 1. F.normalize -> [0,1] -> lerp
        
        # rotate normal!
        normal_final = normals
        V, _, H, W = normal_final.shape # [1, 3, h, w]
        normal_final = (transform[:3, :3].unsqueeze(0) @ normal_final.permute(0, 2, 3, 1).reshape(-1, 3, 1)).reshape(V, H, W, 3).permute(0, 3, 1, 2).contiguous()

    
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
        results['normals_output'] = normal_final
        results['depths_output'] = F.interpolate(depths.unsqueeze(1), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]

        

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
        
        
        
        # debug = False
        # if debug:
        #     # sample = torch.load("/home/xuyimeng/Repo/LaRa/sample.pth")
        #     # print("before",cam_poses.shape)
        #     # cam_poses = sample["tar_c2w"][0][:len(cam_poses)].to(cam_poses.device)
        #     # print("after", cam_poses.shape)
        #     # print("Loading from sample")
        #     self.proj_matrix = torch.tensor([[ 2.7776,  0.0000,  0.0000,  0.0000],
        #             [ 0.0000,  2.7776,  0.0000,  0.0000],
        #             [ 0.0000,  0.0000,  1.4749,  1.0000],
        #             [ 0.0000,  0.0000, -1.1205,  0.0000]]).to(cam_poses.device)
    
            
        # should use the same world coord as gs renderer to convert depth into world_xyz
        results['c2w_colmap'] = cam_poses[:self.opt.num_input_views].clone() 
        
        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4], w2c
        cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4], w2pix
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        
        # print(len(cam_pos))
        
        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos

        # results['vids'] = torch.tensor(final_vids)
        results['vids'] = final_vids
        results['scene_name'] = uid.split('/')[-1]
        

        return results