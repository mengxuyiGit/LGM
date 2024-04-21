import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import kiui
from kiui.lpips import LPIPS

from core.unet import UNet
from core.options import Options
from core.gs import GaussianRenderer

from ipdb import set_trace as st
import time
import einops
import glob
import os
import math

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


class LGM(nn.Module):
    def __init__(
        self,
        opt: Options,
    ):
        super().__init__()

        self.opt = opt

        # unet
        self.unet = UNet(
            9, 14, 
            down_channels=self.opt.down_channels,
            down_attention=self.opt.down_attention,
            mid_attention=self.opt.mid_attention,
            up_channels=self.opt.up_channels,
            up_attention=self.opt.up_attention,
        )

        # last conv
        self.conv = nn.Conv2d(14, 14, kernel_size=1) # NOTE: maybe remove it if train again

        # Gaussian Renderer
        self.gs = GaussianRenderer(opt)

        # activations...
        self.pos_act = lambda x: x.clamp(-1, 1)
        self.scale_act = lambda x: 0.1 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = F.normalize # NOTE: assume the network directly output the world space rotation in quaternion. Only change to axis angle during encoder decoder.
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5 # NOTE: may use sigmoid if train again

        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)
        
        # Create an nn.Parameter for the output
        # self.output_parameter = nn.Parameter(torch.randn((your_output_shape_here), requires_grad=True))
        self.splatter_out = nn.Parameter(torch.randn((1, 6, 14, self.opt.splat_size, self.opt.splat_size), requires_grad=True))
        self.splatter_out_is_random=True


        if self.opt.use_splatter_with_depth_offset:
            self.init_ray_dirs()
            # self.depth_act = nn.Sigmoid()
            self.depth_act = lambda x: x
        else:
            print("We are not using depth and offset")
    

    # ------- depth + offset helper funcs [begin] ------- 

    def init_ray_dirs_legacy(self):
        x = torch.linspace(-self.cfg.data.training_resolution // 2 + 0.5, 
                            self.cfg.data.training_resolution // 2 - 0.5, 
                            self.cfg.data.training_resolution) 
        y = torch.linspace( self.cfg.data.training_resolution // 2 - 0.5, 
                           -self.cfg.data.training_resolution // 2 + 0.5, 
                            self.cfg.data.training_resolution)
        if self.cfg.model.inverted_x:
            x = -x
        if self.cfg.model.inverted_y:
            y = -y
        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        ones = torch.ones_like(grid_x, dtype=grid_x.dtype)
        ray_dirs = torch.stack([grid_x, grid_y, ones]).unsqueeze(0)

        # for cars and chairs the focal length is fixed across dataset
        # so we can preprocess it
        # for co3d this is done on the fly
        if self.cfg.data.category == "cars" or self.cfg.data.category == "chairs" \
            or self.cfg.data.category == "objaverse":
            ray_dirs[:, :2, ...] /= fov2focal(self.cfg.data.fov * np.pi / 180, 
                                              self.cfg.data.training_resolution)
        self.register_buffer('ray_dirs', ray_dirs)
    
    def init_ray_dirs(self):

        # res = self.opt.output_size # TODO: confirm using this dim
        res = self.opt.splat_size

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
        
        focal = res * 0.5 / math.tan(0.5 * np.deg2rad(self.opt.fovy)) # or change to torch.deg2tan
        ray_dirs[:, :2, ...] /= focal
        self.register_buffer('ray_dirs', ray_dirs) # [1 3 128 128]


        ray_dists_sq = torch.sum(ray_dirs**2, dim=1, keepdim=True)
        ray_dists = ray_dists_sq.sqrt() # [1 1 128 128]
        self.register_buffer('ray_dists', ray_dists) # [1 3 128 128]

    def get_pos_from_network_output(self, depth_network, offset, focals_pixels=None, const_offset=None):
        '''
        depth_network.shape: [V, HW, 1]
        offset.shape: [V, HW, 2 or 3]
        '''
        if offset.shape[-1] == 2:
            offset = torch.cat([offset, torch.zeros_like(offset[...,-1:])], dim=-1)

        # expands ray dirs along the batch dimension
        # adjust ray directions according to fov if not done already
        ray_dirs_xy = self.ray_dirs.expand(depth_network.shape[0], 3, *self.ray_dirs.shape[2:])
        # if self.cfg.data.category != "cars" and self.cfg.data.category != "chairs":
        #     assert torch.all(focals_pixels > 0)
        #     ray_dirs_xy = ray_dirs_xy.clone()
        #     ray_dirs_xy[:, :2, ...] = ray_dirs_xy[:, :2, ...] / focals_pixels.unsqueeze(2).unsqueeze(3)

        # depth and offsets are shaped as (b 3 h w)
        if const_offset is not None:
            depth = self.depth_act(depth_network) * (self.opt.zfar - self.opt.znear) + self.opt.znear + const_offset
        else:
            depth = self.depth_act(depth_network) * (self.opt.zfar - self.opt.znear) + self.opt.znear

        ray_dirs_xy = einops.rearrange(ray_dirs_xy, 'v c h w -> v (h w) c') # where c=3
        print(f"depth={depth.shape}, ray_dirs_xy={ray_dirs_xy.shape}, offset={offset.shape}")
       
        pos = ray_dirs_xy * depth + offset # v h w c

        return pos # this is still in cam space


    ##  ------- depth + offset helper funcs [end] ------- 
            

    def clear_splatter_out(self):
        self.splatter_out_is_random=True


    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict


    def prepare_default_rays(self, device, elevation=0):
        
        from kiui.cam import orbit_camera
        from core.utils import get_rays

        cam_poses = np.stack([
            orbit_camera(elevation, 0, radius=self.opt.cam_radius),
            orbit_camera(elevation, 90, radius=self.opt.cam_radius),
            orbit_camera(elevation, 180, radius=self.opt.cam_radius),
            orbit_camera(elevation, 270, radius=self.opt.cam_radius),
        ], axis=0) # [4, 4, 4]
        cam_poses = torch.from_numpy(cam_poses)

        rays_embeddings = []
        for i in range(cam_poses.shape[0]):
            rays_o, rays_d = get_rays(cam_poses[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

            ## visualize rays for plotting figure
            # kiui.vis.plot_image(rays_d * 0.5 + 0.5, save=True)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous().to(device) # [V, 6, h, w]
        
        return rays_embeddings
    
    def get_depth_offset_from_x(self, x, data):

        B, V, C, H, W = x.shape

        # w2c 
        w2c = torch.empty_like(data['c2w_colmap'])

        B_cam, V_cam, _, _ = data['c2w_colmap'].shape  # Get the batch size and number of views
        assert B_cam==B and V_cam==V
        # Iterate over each sample in the batch and each view
        for b in range(B_cam):
            for v in range(V_cam):
                # Invert each matrix individually
                w2c[b, v] = data['c2w_colmap'][b, v].inverse()

        # If you need to transpose the last two dimensions, you can do it after the loop
        w2c = w2c.transpose(-2, -1)  # Transpose the last two dimensions
        w2c = einops.rearrange(w2c, 'b v row col -> (b v) row col') # [B*V, 4, 4]


        world_xyz = einops.rearrange(x[:, :, :3, ...], 'b v c h w -> (b v) (h w) c')
        world_xyz_homo = torch.cat([world_xyz, torch.ones_like(world_xyz[...,0:1])], dim=-1)

        cam_xyz_homo = torch.bmm(world_xyz_homo, w2c)
        cam_xyz = cam_xyz_homo[...,:3] / cam_xyz_homo[...,3:]

        # cam xyz -> cam view depth + cam view offset
        # depth: not z, but the absolute number of rays

        # offset: cam_xyz - depth * ray_dirs 
        # NOTE: self.ray_dirs is not normalized
        
        
        cam_xyz_dists_sq = torch.sum(cam_xyz**2, dim=-1, keepdim=True)
        # print("cam_xyz_dists_sq.shape", cam_xyz_dists_sq.shape)
    
        cam_xyz_dists = cam_xyz_dists_sq.sqrt()
        batch_ray_dists = einops.rearrange(self.ray_dists, 'b c h w -> b (h w) c').repeat(B*V, 1, 1)
        # print(batch_ray_dists.shape)
        
        depth = cam_xyz_dists / batch_ray_dists # expect [bv hw 1]


        batch_ray_dirs = einops.rearrange(self.ray_dirs, 'b c h w -> b (h w) c').repeat(B*V, 1, 1)
        cam_xyz_from_depth = batch_ray_dirs * depth # expect [bv hw 3]

        xyz_offset = cam_xyz - cam_xyz_from_depth

        # the return should be in shape of x
        depth = einops.rearrange(depth, '(b v) (h w) c -> b v c h w', b=B, v=V, h=H, w=W)
        xyz_offset = einops.rearrange(xyz_offset, '(b v) (h w) c -> b v c h w', b=B, v=V, h=H, w=W)

        return depth, xyz_offset
    

    def get_organized_depth_offset_and_x_from_x(self, x, data):

        B, V, C, H, W = x.shape

        new_x = x.clone()

        # w2c 
        w2c = torch.empty_like(data['c2w_colmap'])

        B_cam, V_cam, _, _ = data['c2w_colmap'].shape  # Get the batch size and number of views
        assert B_cam==B and V_cam==V
        # Iterate over each sample in the batch and each view
        for b in range(B_cam):
            for v in range(V_cam):
                # Invert each matrix individually
                w2c[b, v] = data['c2w_colmap'][b, v].inverse()

        # If you need to transpose the last two dimensions, you can do it after the loop
        w2c = w2c.transpose(-2, -1)  # Transpose the last two dimensions
        w2c = einops.rearrange(w2c, 'b v row col -> (b v) row col') # [B*V, 4, 4]


        world_xyz = einops.rearrange(x[:, :, :3, ...], 'b v c h w -> (b v) (h w) c')
        world_xyz_homo = torch.cat([world_xyz, torch.ones_like(world_xyz[...,0:1])], dim=-1)

        cam_xyz_homo = torch.bmm(world_xyz_homo, w2c)
        cam_xyz = cam_xyz_homo[...,:3] / cam_xyz_homo[...,3:]

        # st()
        ## reorganize splatter image 
        # Apply perspective division
        # cam_xyz[..., :2]: -0.6~+0.6, cam_xyz[..., 2:]: 0.99 ~ 1.98
        projected_coords = cam_xyz[..., :2] / cam_xyz[..., 2:] # projected_coords: -0.5 ~ +0.5
        
        # TODO: check whether the input range should be in [-1,1] or [-0.5,0.5]
        projected_coords = (projected_coords + 1) * torch.tensor([W,H], device=cam_xyz.device)[None, None, :] / 2 

        MAX_DEPTH = cam_xyz[..., 2:].max() * 2
        organized_depth = torch.full((projected_coords.shape[0], H, W), MAX_DEPTH, dtype=x.dtype, device=x.device) # [bv, H, W]

        for _bv in range(projected_coords.shape[0]):
            for i in range(projected_coords.shape[1]):
                u, v = projected_coords[_bv, i].round().int()

                if 0 <= u < W and 0 <= v < H:
                    # depth = (2 * self.z_far * self.z_near) / (self.z_far + self.z_near - projected_coords[i, 2] * (self.z_far - self.z_near))
                    organized_depth[_bv, v, u] = min(cam_xyz[_bv, i, 2], organized_depth[_bv, v, u])
                    # TODO: also reorganize everything else in x. Now we only visualize depth to see the effect

        
        # st() # should acquire an organized depth map in the shape of [_bv, H, W]
        organized_depth = organized_depth.unsqueeze(-1) # add a C dim at last

        organized_depth_to_vis = einops.rearrange(organized_depth, "n h w c ->  h (n w) c", h=H, w=W)
        organized_depth_to_vis = (organized_depth_to_vis - organized_depth_to_vis.min()) / MAX_DEPTH
        kiui.write_image("organized_depth.png", organized_depth_to_vis.to(torch.float16))
        



        # # def depth_of_object(self):
        # # Transform object coordinates to camera coordinates
        # object_coords_homogeneous = torch.cat((self.verts, torch.ones((self.verts.shape[0], 1), dtype=torch.float32, device='cuda')), dim=1)

        # # # Apply perspective projection
        # object_coords_camera = torch.matmul(object_coords_homogeneous, self.viewpoint_camera2.full_proj_transform)

        # # Apply perspective division
        # projected_coords = object_coords_camera[:, :2] / object_coords_camera[:, 2:]

        # Normalize points to image coordinates
        # projected_coords = (projected_coords + 1) * torch.tensor([self.viewpoint_camera2.image_width, self.viewpoint_camera2.image_height], device='cuda')[None, :] / 2

        # Initialize depth image with maximum depth value
        # depth_image = torch.full((self.viewpoint_camera2.image_height, self.viewpoint_camera2.image_width), self.z_far, dtype=torch.float32, device='cuda')
        # Calculate depth for each projected point and update depth image
        # for i in range(self.verts.shape[0]):
        #     u, v = projected_coords[i, :2].round().int()
        #     if 0 <= u < self.viewpoint_camera2.image_width and 0 <= v < self.viewpoint_camera2.image_height:
        #         # depth = (2 * self.z_far * self.z_near) / (self.z_far + self.z_near - projected_coords[i, 2] * (self.z_far - self.z_near))
        #         depth_image[v, u] = min(object_coords_camera[i, 2], depth_image[v, u])
        # self.depth_image = depth_image.reshape(-1, 1)



        # cam xyz -> cam view depth + cam view offset
        # depth: not z, but the absolute number of rays

        # offset: cam_xyz - depth * ray_dirs 
        # NOTE: self.ray_dirs is not normalized
        
        
        cam_xyz_dists_sq = torch.sum(cam_xyz**2, dim=-1, keepdim=True)
        # print("cam_xyz_dists_sq.shape", cam_xyz_dists_sq.shape)
    
        cam_xyz_dists = cam_xyz_dists_sq.sqrt()
        batch_ray_dists = einops.rearrange(self.ray_dists, 'b c h w -> b (h w) c').repeat(B*V, 1, 1)
        # print(batch_ray_dists.shape)
        
        depth = cam_xyz_dists / batch_ray_dists # expect [bv hw 1]
        
        unorganized_depth_to_vis = einops.rearrange(depth, 'n (h w) c -> h (n w) c', h=H, w=W)
        unorganized_depth_to_vis = (unorganized_depth_to_vis - unorganized_depth_to_vis.min()) / unorganized_depth_to_vis.max()
        kiui.write_image("unorganized_depth.png", unorganized_depth_to_vis.to(torch.float16))

        st()


        batch_ray_dirs = einops.rearrange(self.ray_dirs, 'b c h w -> b (h w) c').repeat(B*V, 1, 1)
        cam_xyz_from_depth = batch_ray_dirs * depth # expect [bv hw 3]

        xyz_offset = cam_xyz - cam_xyz_from_depth

        # the return should be in shape of x
        depth = einops.rearrange(depth, '(b v) (h w) c -> b v c h w', b=B, v=V, h=H, w=W)
        xyz_offset = einops.rearrange(xyz_offset, '(b v) (h w) c -> b v c h w', b=B, v=V, h=H, w=W)

        assert new_x.shape == x.shape
        assert not torch.allclose(new_x, x) # new_x and x should not be the same tensor

        return depth, xyz_offset, new_x 
    

    def get_world_xyz_from_depth_offset(self, depth_activated, xyz_offset, data):

        B, V, C, H, W = depth_activated.shape

        depth = einops.rearrange(depth_activated, 'b v c h w -> (b v) (h w) c', b=B, v=V, h=H, w=W)
        xyz_offset = einops.rearrange(xyz_offset, 'b v c h w -> (b v) (h w) c', b=B, v=V, h=H, w=W)

        batch_ray_dirs = einops.rearrange(self.ray_dirs, 'b c h w -> b (h w) c').repeat(depth.shape[0], 1, 1)

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


    def get_depth_offset_from_world_xyz_v1(self, world_xyz, data):


        B, V, C, H, W = world_xyz.shape
        world_xyz = einops.rearrange(world_xyz, 'b v c h w -> (b v) (h w) c') # under c1 view
                
        xyz_homo = torch.cat([world_xyz, torch.ones_like(world_xyz[...,0:1])], dim=-1)
        
        # process cam matrix of c1 -> c

        # NOTE: should inverse each cam one by one, now is the w2c of B*V
        B_cam, V_cam, _, _ = data['c2w_colmap'].shape  # Get the batch size and number of views
        print(f"B_cam={B_cam}, B={B} and V_cam={V_cam} V={V}")
        assert B_cam==B and V_cam==V
        w2c = torch.empty_like(data['c2w_colmap'])  # Preallocate the w2c tensor

        # Iterate over each sample in the batch and each view
        for b in range(B_cam):
            for v in range(V_cam):
                # Invert each matrix individually
                w2c[b, v] = data['c2w_colmap'][b, v].inverse()

        # If you need to transpose the last two dimensions, you can do it after the loop
        w2c = w2c.transpose(-2, -1)  # Transpose the last two dimensions
        w2c = einops.rearrange(w2c, 'b v row col -> (b v) row col') # [B*V, 4, 4]
        

        # begin transformation from cam
        print("w2c.shape, xyz.shape", w2c.shape, world_xyz.shape)

        cam_xyz_homo = torch.bmm(xyz_homo, w2c) # where the batch_dim = B*V
        cam_xyz = cam_xyz_homo[...,:3] / cam_xyz_homo[...,3:]
        
        depth = cam_xyz[...,2:3] # NOTE: this is not a strict depth, but z. However, this is only a course init, therefore not that important.

        debug = True
        if debug : 
            # save a vis of depth 
            depth_vis = einops.rearrange(depth, '(b v) (h w) c-> b v c h w', b=B, v=V, h=H, w=W)
            depth_vis = depth_vis.to(torch.float32)
            # depth_vis = depth_vis.max() - depth_vis

            depth_vis_save = depth_vis.detach().cpu().numpy() # [B, V, 3, output_size, output_size]
            depth_vis_save = depth_vis_save.transpose(0, 3, 1, 4, 2).reshape(-1, depth_vis_save.shape[1] * depth_vis_save.shape[3], 1) # [B*output_size, V*output_size, 3]
            
            kiui.write_image(f'{self.opt.workspace}/depth_from_init_z.jpg', depth_vis_save)
            kiui.write_image(f'{self.opt.workspace}/all_zeros.jpg', np.zeros_like(depth_vis_save))
            kiui.write_image(f'{self.opt.workspace}/all_ones.jpg', np.ones_like(depth_vis_save))


            
            z_vis = world_xyz[...,-1:]
            # FIXME: require normalization?
            sp_min, sp_max =  -0.7, 0.7
            z_vis = (z_vis - sp_min) / (sp_max - sp_min)
            z_vis = einops.rearrange(z_vis, '(b v) (h w) c -> b v c h w', b=B, v=V, h=H, w=W)
            z_vis = z_vis.to(torch.float32)

            z_vis_save = z_vis.detach().cpu().numpy() # [B, V, 3, output_size, output_size]
            z_vis_save = z_vis_save.transpose(0, 3, 1, 4, 2).reshape(-1, z_vis_save.shape[1] * z_vis_save.shape[3], 1) # [B*output_size, V*output_size, 3]
            
            kiui.write_image(f'{self.opt.workspace}/init_z.jpg', z_vis_save)


        cam_xy =  cam_xyz[...,:2]

        if self.opt.zero_init_xy_offset or self.opt.always_zero_xy_offset:
            # set offset to 0.0
            print("You choose to init xy_offset as all zeros")
            xy_offset = torch.zeros_like(cam_xy)

            depth = einops.rearrange(depth, '(b v) (h w) c-> b v c h w', v=V, h=H, w=W)
            xy_offset = einops.rearrange(xy_offset, '(b v) (h w) c-> b v c h w', v=V, h=H, w=W)   
        
        else:
            # calculate xy_offset
            # depth + offset -> cam xyz: by homography

            # loop over batch, because the get_pos_from_network_output takes view as batch dim due to the self.ray_dirs
            '''
                depth_network.shape: [V, HW, 1]
                offset.shape: [V, HW, 2 or 3]
            '''
            depth = einops.rearrange(depth, '(b v) hw c-> b v hw c', v=V)
            
            cam_xy = einops.rearrange(cam_xy, '(b v) hw c-> b v hw c', v=V)  # xyz offset under cam view

            xy_offset_list = []

            focals_pixels = None
            const_offset = None # float. assume no const offset for now TODO


            for b in range(B):
                _depth = depth[b] # [v hw c]
                _zero_offset = torch.zeros_like(cam_xy[b])
                
                cam_pos_no_offset = self.get_pos_from_network_output(_depth, _zero_offset, focals_pixels=focals_pixels, const_offset=const_offset) # [v n 3]
                cam_xy_anchor = cam_pos_no_offset[..., :2]

                _cam_xy = cam_xy[b]
                _xy_offset = _cam_xy - cam_xy_anchor
                xy_offset_list.append(_xy_offset[None])

            xy_offset = torch.cat(xy_offset_list, dim=0)
            
            depth = einops.rearrange(depth, 'b v (h w) c -> b v c h w', v=V, h=H, w=W)
            xy_offset = einops.rearrange(xy_offset, 'b v (h w) c -> b v c h w', v=V, h=H, w=W)   
        
        return {'depth':depth, 'xy_offset': xy_offset}
        

    def forward_gaussians(self, images, data=None):
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]

        B, V, C, H, W = images.shape
        
        ### discarded all the before modules, direcly load from the nnParameters
        ### ----- previous -------
        # images = images.view(B*V, C, H, W)

        # x = self.unet(images) # [B*4, 14, h, w]
        # x = self.conv(x) # [B*4, 14, h, w]
        
        # x = x.reshape(B, 4, 14, self.opt.splat_size, self.opt.splat_size)
        
        ### ----- new -------
        if self.splatter_out_is_random: # world xyz --> depth + offset
            
            # run LGM
            images = images.view(B*V, C, H, W)

            x = self.unet(images) # [B*4, 14, h, w]
            x = self.conv(x) # [B*4, 14, h, w]
            
            x = x.reshape(B, 6, 14, self.opt.splat_size, self.opt.splat_size)
            ## assign the pretrained output to spaltter out

            # update the C,H,W to be that of splatter image
            B, V, C, H, W = x.shape

            debug = False
            if debug: 
                raise NotImplementedError 

                depth, xyz_offset = self.get_depth_offset_from_x(x, data=data)
                
                depth_activated = self.depth_act(depth)
                world_xyz2 = self.get_world_xyz_from_depth_offset(depth_activated, xyz_offset, data=data)
                world_xyz2 = einops.rearrange(world_xyz2, 'b (v h w) c -> b v c h w', b=B, v=V, h=H, w=W)
            
                # assert torch.allclose(world_xyz, world_xyz2)
                # print("torch.allclose(world_xyz, world_xyz2)",torch.allclose(world_xyz, world_xyz2)) # NOTE: they are not closein numerical value, but the rendering result seems correct
                # st()

                # cat with x
                x = torch.cat([world_xyz2, x[:,:,3:,...]], dim=2)


            elif self.opt.use_splatter_with_depth_offset:
               
                assert data is not None, "Require the data['c2w'] to convert depth to xyz"
                if self.opt.reorganize_splatter_init:
                    depth, xyz_offset, x = self.get_organized_depth_offset_and_x_from_x(x, data=data)
                    # NOTE: the x has to be reorganized to be consistent with the depth and offset
                
                else:
                    depth, xyz_offset = self.get_depth_offset_from_x(x, data=data)
                
                
                if self.opt.always_zero_xy_offset:
                    xyz_offset = torch.zeros_like(xyz_offset)
                
                # NOTE: depth is at the last dim
                # NOTE: this depth should not be activated
                x = torch.cat([xyz_offset, x[:,:,3:,...], depth], dim=2)


            
            self.splatter_out = nn.Parameter(x) # now the parameters are depth and offset
            # print("Init gaussian parameters of shape:", x.shape)
         
            ## toggle the flag
            self.splatter_out_is_random = False
            print("Only do this once: change random init to pretrained output")
            
        
        
        assert B==1 #TODO: can we handle multiple optimization in one loop?
        x = self.splatter_out
        ### ----- new end -------
        
        ## visualize multi-view gaussian features for plotting figure
        # tmp_alpha = self.opacity_act(x[, :, 3:4])
        # tmp_img_rgb = self.rgb_act(x[0, :, 11:]) * tmp_alpha + (1 - tmp_alpha)
        # tmp_img_pos = self.pos_act(x[0, :, 0:3]) * 0.5 + 0.5
        # kiui.vis.plot_image(tmp_img_rgb, save=True)
        # kiui.vis.plot_image(tmp_img_pos, save=True)

        gaussians = self.get_activated_splatter_out(data=data)
        
        # fuse all views
        gaussians = einops.rearrange(gaussians, 'b v n c -> b (v n) c')

        return gaussians
    
    
    def get_activated_splatter_out(self, data=None):

        B, V, C, H, W, = self.splatter_out.shape

        x = einops.rearrange(self.splatter_out, 'b v c h w -> b (v h w) c')
       
        if self.opt.use_splatter_with_depth_offset:

            assert (C == 15) and (data != None)

            xyz_offset = self.splatter_out[:, :, 0:3,...]
            depth = self.splatter_out[:, :, 14:,...] # NOTE: append the depth to the last dim
            
            # FIXME: we only allow zero offset now, to better align with the shape of splatter 
            # print("FIXME: we only allow zero offset now, to better align with the shape of splatter ")
            if self.opt.always_zero_xy_offset:
                xyz_offset = torch.zeros_like(xyz_offset)

            # depth act
            depth_activated = self.depth_act(depth)

            world_xyz2 = self.get_world_xyz_from_depth_offset(depth_activated, xyz_offset, data=data)
         
            pos = self.pos_act(world_xyz2) # [B, N, 3]: depth + offset / world xyz -> world xyz
          
        else:
            pos = self.pos_act(x[...,:3])
        
        # same as before, except explicitly end the dim of rgbs
         #  # x = einops.rearrange(self.splatter_out, 'b v c h w -> b v (h w) c') --> NOTE:Will result in error!!!

        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11]) 
        rgbs = self.rgb_act(x[..., 11:14]) # NOTE: explicit end rgbs

        spaltter_batch = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, 4, N, 14]
        spaltter_batch = einops.rearrange(spaltter_batch, 'b (v h w) c -> b v (h w) c', v=V, h=H, w=W)

        return spaltter_batch
    
    def get_raw_splatter_out(self,):
        assert not self.splatter_out_is_random, "Random init splatter out should not be saved"

        return self.splatter_out
    
    def forward(self, data, step_ratio=1., opt=None, epoch=-1, i=-1):
        # data: output of the dataloader
        # return: loss

        results = {}
        loss = 0

        images = data['input'] # [B, 4, 9, h, W], input features
        

        # use the first view to predict gaussians
        gaussians = self.forward_gaussians(images, data=data) # [B, N, 14]


        # random bg for training
        if self.training:
            bg_color = torch.rand(3, dtype=torch.float32, device=gaussians.device)
        else:
            bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)

        # use the other views for rendering and supervision
        results = self.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
        # st()
        pred_images = results['image'] # [B, V, C, output_size, output_size]
        pred_alphas = results['alpha'] # [B, V, 1, output_size, output_size]
        
        ## also output gaussians
        results['gaussians'] = gaussians
        
        # print(f"-------3. after gs.render:{time.time()-last_time}---------")
        # last_time = time.time()

        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas

        gt_images = data['images_output'] # [B, V, 3, output_size, output_size], ground-truth novel views
        gt_masks = data['masks_output'] # [B, V, 1, output_size, output_size], ground-truth masks
        ## V=8
        
        gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)
        

        loss_mse = F.mse_loss(pred_images, gt_images) + F.mse_loss(pred_alphas, gt_masks)
        loss = loss + loss_mse
        
        # print('train vids',[t.item() for t in data['vids']])
        if (opt is not None) and (epoch % opt.save_train_pred == 0) and epoch > 0:
            
            ### ----------- debug-------------
            gt_images_save = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
            gt_images_save = gt_images_save.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images_save.shape[1] * gt_images_save.shape[3], 3) # [B*output_size, V*output_size, 3]
            kiui.write_image(f'{opt.workspace}/train_gt_images_{epoch}_{i}.jpg', gt_images_save)

            pred_images_save = results['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
            pred_images_save = pred_images_save.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images_save.shape[1] * pred_images_save.shape[3], 3)
            kiui.write_image(f'{opt.workspace}/train_pred_images_{epoch}_{i}.jpg', pred_images_save)
            ### -------- debug [end]-------------

        if self.opt.lambda_lpips > 0:
            loss_lpips = self.lpips_loss(
                # gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                # pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                # downsampled to at most 256 to reduce memory cost
                F.interpolate(gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
                F.interpolate(pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            ).mean()
            results['loss_lpips'] = loss_lpips
            loss = loss + self.opt.lambda_lpips * loss_lpips
            
        results['loss'] = loss

        # metric
        with torch.no_grad():
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
            results['psnr'] = psnr
        
        # print(f"-------4. after metric:{time.time()-last_time}---------")

        return results