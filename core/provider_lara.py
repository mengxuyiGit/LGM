import numpy as np
from glob import glob
import random
import torch
# from dataLoader.utils import build_rays
from scipy.spatial.transform import Rotation as R

import h5py

from core.options import Options
from ipdb import set_trace as st
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from core.utils import get_rays, grid_distortion, orbit_camera_jitter

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def fov_to_ixt(fov, reso):
    ixt = np.eye(3, dtype=np.float32)
    ixt[0][2], ixt[1][2] = reso[0]/2, reso[1]/2
    focal = .5 * reso / np.tan(.5 * fov)
    ixt[[0,1],[0,1]] = focal
    return ixt

class gobjverse(torch.utils.data.Dataset):
    # def __init__(self, cfg):
    def __init__(self, opt: Options, name=None, training=True):
        super(gobjverse, self).__init__()
        self.cfg = opt
        self.opt = opt
        self.data_root = opt.data_path_rendering
        
        
        self.split = 'train' if training else 'test'
        self.training = training
        self.img_size = np.array([512]*2)

        self.metas = h5py.File(self.data_root, 'r')
        print("Loading data from", self.data_root)
        print("Number of scenes", len(self.metas.keys()))
        scenes_name = np.array(sorted(self.metas.keys())) # [:1000]
        
        
        if 'splits' in scenes_name:
            self.scenes_name = self.metas['splits']['test'][:].astype(str) #self.metas['splits'][self.split]
        else:
            n_scenes = 300000
            i_test = np.arange(len(scenes_name))[::10][:10] # only test 10 scenes
            i_train = np.array([i for i in np.arange(len(scenes_name)) if
                            (i not in i_test)])[:n_scenes]
            if opt.overfit_one_scene:
                i_test = [0]
                i_train = i_test*1000
            self.scenes_name = scenes_name[i_train] if self.split=='train' else scenes_name[i_test]
            print("Number of scenes", len(self.scenes_name))
            
        self.b2c = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.n_group = 4 # cfg.n_group
        
        self.cfg.load_normal = True
            
        # default camera intrinsics
        assert opt.fovy == 39.6
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1
       
    
    def __getitem__(self, index):
    
        scene_name = self.scenes_name[index]
        # print("scene_name", scene_name)
        scene_info = self.metas[scene_name]

        if self.split=='train' and self.n_group > 1:
            # print("111")
            src_view_id = [random.choices(scene_info['groups'][f'groups_{self.n_group}_{i}'])[0] for i in torch.randperm(self.n_group).tolist()]
            view_id = src_view_id + [random.choices(scene_info['groups'][f'groups_{self.n_group}_{i}'])[0] for i in torch.randperm(self.n_group).tolist()]
        elif self.n_group == 1:
            # print("222")
            src_view_id = [scene_info['groups'][f'groups_4_{i}'][0] for i in range(1)]
            view_id = src_view_id + [scene_info['groups'][f'groups_4_{i}'][-1] for i in range(4)]
        else:
            # print("333")
            src_view_id = [scene_info['groups'][f'groups_{self.n_group}_{i}'][0] for i in range(self.n_group)]
            view_id = src_view_id + [scene_info['groups'][f'groups_4_{i}'][-1] for i in range(4)]
        
        fixed_input_views = np.arange(25, 37)[::3].tolist() + [2,22] # equals to the original GOBjaverse 27, 30, 33, 36, 2, 22 (because h5 do not include the 25,26 views)
        view_id = fixed_input_views + np.random.permutation(np.arange(0,38))[:(self.opt.num_views-self.opt.num_input_views)].tolist()
        # print("view_id", len(view_id))

        tar_img, bg_colors, tar_nrms, tar_msks, tar_c2ws, tar_w2cs, tar_ixts = self.read_views(scene_info, view_id, scene_name)

        results = {}
    
        images = torch.from_numpy(tar_img).permute(0,3,1,2) # [V, C, H, W]
        normals = torch.from_numpy(tar_nrms).permute(0,3,1,2) # [V, C, H, W]
        # depths = tar_img #[TODO: lara processed data has no depth]
        masks = torch.from_numpy(tar_msks).to(images.dtype) #.unsqueeze(1) # [V, C, H, W]
        cam_poses = torch.from_numpy(tar_c2ws)
        

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
        V, _, H, W = normal_final.shape # [1, h, w, 3]
    
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

        results['scene_name'] = scene_name #uid.split('/')[-1]
        

        return results
    
    
    def read_views(self, scene, src_views, scene_name):
        src_ids = src_views
        bg_colors = []
        ixts, exts, w2cs, imgs, msks, normals = [], [], [], [], [], []
        for i, idx in enumerate(src_ids):
            
            # if self.split!='train' or i < self.n_group:
            #     bg_color = np.ones(3).astype(np.float32)
            # else:
            #     bg_color = np.ones(3).astype(np.float32)*random.choice([0.0, 0.5, 1.0])
            bg_color = np.ones(3).astype(np.float32)

            bg_colors.append(bg_color)
            
            img, normal, mask = self.read_image(scene, idx, bg_color, scene_name)
            imgs.append(img)
            ixt, ext, w2c = self.read_cam(scene, idx)
            ixts.append(ixt)
            exts.append(ext)
            w2cs.append(w2c)
            msks.append(mask)
            normals.append(normal)
        return np.stack(imgs), np.stack(bg_colors), np.stack(normals), np.stack(msks), np.stack(exts), np.stack(w2cs), np.stack(ixts)

    def read_cam(self, scene, view_idx):
        c2w = np.array(scene[f'c2w_{view_idx}'], dtype=np.float32)
        # print("c2w", c2w.shape, "view_idx", view_idx)   
        lara = False
        if not lara:
            # blender world + opencv cam --> opengl world & cam
            c2w[1] *= -1
            c2w[[1, 2]] = c2w[[2, 1]]
            c2w[:3, 1:3] *= -1 # invert up and forward direction
            
        w2c = np.linalg.inv(c2w)
        fov = np.array(scene[f'fov_{view_idx}'], dtype=np.float32)
        ixt = fov_to_ixt(fov, self.img_size)
        return ixt, c2w, w2c

    def read_image(self, scene, view_idx, bg_color, scene_name):
        
        img = np.array(scene[f'image_{view_idx}'])

        mask = (img[...,-1] > 0).astype('uint8')
        img = img.astype(np.float32) / 255.
        img = (img[..., :3] * img[..., -1:] + bg_color*(1 - img[..., -1:])).astype(np.float32)

        if self.cfg.load_normal:

            normal = np.array(scene[f'normal_{view_idx}'])
            normal = normal.astype(np.float32) / 255. * 2 - 1.0
            normal[...,[0,1]] = -normal[...,[0,1]]
            normal = normal[...,[1,2,0]]
        

            # rectify normal directions
            normal = normal[..., ::-1]
            normal[..., 0] *= -1
            normal = normal * np.expand_dims(mask, axis=-1)  # to [0, 0, 0] bg

            return img, normal, mask

        return img, None, mask


    def __len__(self):
        return len(self.scenes_name)

def get_K_from_params(params):
    K = np.zeros((3, 3)).astype(np.float32)
    K[0][0], K[0][2], K[1][2] = params[:3]
    K[1][1] = K[0][0]
    K[2][2] = 1.
    return K