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
        self.rot_act = F.normalize
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5 # NOTE: may use sigmoid if train again

        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)
        
        # Create an nn.Parameter for the output
        # self.output_parameter = nn.Parameter(torch.randn((your_output_shape_here), requires_grad=True))
        self.splatter_out = nn.Parameter(torch.randn((1, 4, 14, self.opt.splat_size, self.opt.splat_size), requires_grad=True))
        self.splatter_out_is_random=True
    
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
    
    # inverse acts : for load splatter ckpt 
    def inverse_scale_act(self, y):
        # Inverse of y = 0.1 * softplus(x) is x = log(exp(10*y) - 1)
        # Add a small epsilon to prevent log(0)
        epsilon = 1e-6
        return torch.log(torch.exp(y * 10) - 1 + epsilon)

    def inverse_opacity_act(self, y):
        # Inverse of y = sigmoid(x) is x = log(y / (1 - y))
        # Add a small epsilon to prevent division by zero or log(0)
        epsilon = 1e-6
        return torch.log(y / (1 - y + epsilon) + epsilon)

    def inverse_rgb_act(self, y):
        # Inverse of y = 0.5 * tanh(x) + 0.5 is x = atanh(2 * (y - 0.5))
        ## v1 
        return torch.atanh(2 * y - 1)
        ## v2
        # # torch.atanh is not implemented, so we use numpy and then convert back to tensor
        # # Note: numpy works with cpu tensors, so ensure your tensor is on cpu
        # device = y.device
        # y = y.cpu().detach().numpy()
        # x = np.arctanh(2 * (y - 0.5))
        # return torch.tensor(x, dtype=torch.float32, device=device)  # Convert back to tensor and send to original device if needed


    def forward_gaussians(self, images, splatter_ckpt=None):
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
        if self.splatter_out_is_random and splatter_ckpt == None:
            ## ----- previous -------
            images = images.view(B*V, C, H, W)

            x = self.unet(images) # [B*4, 14, h, w]
            x = self.conv(x) # [B*4, 14, h, w]
            
            x = x.reshape(B, 6, 14, self.opt.splat_size, self.opt.splat_size)
            ## assign the pretrained output to spaltter out
            # self.splatter_out = nn.Parameter(x[:,:1,:,::2,::2]) # NOTE: for FFHQ
            # st()
            self.splatter_out = nn.Parameter(x)
            
            ## toggle the flag
            self.splatter_out_is_random = False
            print("Only do this once: change random init to pretrained output")
            # st()
        
        elif self.splatter_out_is_random and splatter_ckpt is not None:
            # splatter_ckpt [1, 6, 14, 128, 128]
            self.splatter_out_is_random = False
            
            # x = splatter_ckpt.permute(0, 1, 3, 4, 2).reshape(B, -1, 14) # [1, 98304, 14
            B, V, C, H, W = splatter_ckpt.shape
            x = einops.rearrange(splatter_ckpt, "b v c h w -> b (v h w) c")

            ## do reverse activation on splatter ckpt
            pos_deact = x[..., 0:3]
            opacity_deact = self.inverse_opacity_act(x[..., 3:4])
            scale_deact = self.inverse_scale_act(x[..., 4:7])
            rotation_deact = x[..., 7:11]
            rgbs_deact = self.inverse_rgb_act(x[..., 11:])

            x = torch.cat([pos_deact, opacity_deact, scale_deact, rotation_deact, rgbs_deact], dim=-1) # [B, N, 14]
            deacted_splatter_ckpt = einops.rearrange(x, " b (v h w) c -> b v c h w", v=V, h=H, w=W)

            self.splatter_out = nn.Parameter(deacted_splatter_ckpt)
            print("Only do this once: change random init to splatter ckpt")



        assert B==1 #TODO: can we handle multiple optimization in one loop?
        x = self.splatter_out
        ### ----- new end -------
        
        ## visualize multi-view gaussian features for plotting figure
        # tmp_alpha = self.opacity_act(x[0, :, 3:4])
        # tmp_img_rgb = self.rgb_act(x[0, :, 11:]) * tmp_alpha + (1 - tmp_alpha)
        # tmp_img_pos = self.pos_act(x[0, :, 0:3]) * 0.5 + 0.5
        # kiui.vis.plot_image(tmp_img_rgb, save=True)
        # kiui.vis.plot_image(tmp_img_pos, save=True)

        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)
        
        pos = self.pos_act(x[..., 0:3]) # [B, N, 3]
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, N, 14]
        
        return gaussians
    
    def get_activated_splatter_out(self):
        B, V, C, H, W, = self.splatter_out.shape
        # x = einops.rearrange(self.splatter_out, 'b v c h w -> b v (h w) c') --> Will result in error!!!FIXME
        x = einops.rearrange(self.splatter_out, 'b v c h w -> b (v h w) c')
      
        
        pos = self.pos_act(x[..., 0:3]) # [B, 4, N, 3]
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        spaltter_batch = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, 4, N, 14]
        spaltter_batch = einops.rearrange(spaltter_batch, 'b (v h w) c -> b v (h w) c', v=V, h=H, w=W)

        return spaltter_batch
    
    def get_activated_splatter_out2(self):
        x = self.splatter_out
        
        x = x.permute(0, 1, 3, 4, 2).reshape(1, -1, 14)
        
        pos = self.pos_act(x[..., 0:3]) # [B, N, 3]
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, N, 14]
        
        return gaussians
        
        
    
    def forward(self, data, step_ratio=1., opt=None, epoch=-1, i=-1):
        # data: output of the dataloader
        # return: loss

        results = {}
        loss = 0

        images = data['input'] # [B, 4, 9, h, W], input features
        
        # print(f"-------1. before forward_gaussians:---------")
        # last_time = time.time()
        

        # use the first view to predict gaussians
        # st() # check whether data contains optimzied splatter images
        splatter_ckpt = data.get('splatters_output', None)
        # print("splatter_ckpt: ", splatter_ckpt)
        gaussians = self.forward_gaussians(images, splatter_ckpt=splatter_ckpt) # [B, N, 14]
        # verify whether the gaussians and the splatter_out are correct

        # ######## debug code #############
        # ## ----- v1 -----
        # sp_images = self.get_activated_splatter_out()
        # B, V, hw, C = sp_images.shape
        # print(f"B, V, hw, W, C:{B, V, hw, C}")
        # guassians_fused = sp_images.reshape(B, -1, C).to(gaussians.device)
       
        # print(f"two are equal: {torch.all(gaussians == guassians_fused)}")
        
        # # # st() # load gt gaussians for render
        # if not self.training:
        #     if self.opt.eval_fused_gt:
        #         st()
        #         print("Load oreexisitng ply")
        #         gaussians = self.gs.load_ply('/home/xuyimeng/Repo/LGM/data/splatter_gt_full/00000-hydrant-eval_pred_gs_6100_0/fused.ply').to(gaussians.device)
        #         gaussians = gaussians.unsqueeze(0)
        #     elif self.opt.eval_splatter_gt:
        #         st()
        #         splatter_uid = '/home/xuyimeng/Repo/LGM/data/splatter_gt_full/00000-hydrant-eval_pred_gs_6100_0'
        #         # splatter_uid = '/home/xuyimeng/Repo/LGM/runs/LGM_optimize_splatter/workspace_debug/00006-hydrant-gt-splat128-inV6-lossV20-lr0.0006/eval_pred_gs_0_0'
        #         print(f"Load splatter gt ply -- wrong!!!! from: {splatter_uid}")
        #         splatter_images_multi_views = []
        #         spaltter_files = glob.glob(os.path.join(splatter_uid, "splatter_*.ply")) # TODO: make this expresion more precise: only load the ply files with numbers
        #         for sf in spaltter_files:
        #             splatter_im = self.gs.load_ply(sf)
        #             splatter_images_multi_views.append(splatter_im)
                
        #         splatter_images_mv = torch.stack(splatter_images_multi_views, dim=0) # # [6, 16384, 14]
        #         # splatter_res = int(math.sqrt(splatter_images_mv.shape[-2]))
        #         splatter_res = self.opt.splat_size
        #         # print(f"splatter_res: {splatter_res}")
        #         ## when saving the splatter image in model_fix_pretrained.py: x = einops.rearrange(self.splatter_out, 'b v c h w -> b v (h w) c')
        #         splatter_images_mv = einops.rearrange(splatter_images_mv, 'v (h w) c -> v c h w', h=splatter_res, w=splatter_res)

        #         gaussians = splatter_images_mv.unsqueeze(0).permute(0, 1, 3, 4, 2).reshape(1, -1, 14).to(gaussians.device)
        # ######## debug code [END] #############
        
        
        
        
        # print(f"-------2. after forward_gaussians:{time.time()-last_time}---------")
        # last_time = time.time()

        # results['gaussians'] = gaussians #FIXME: WHY do this? results is overwritten

        # random bg for training
        if self.training and not torch.all(data['masks_output']==1):
            st() # This should not st() for srn data, which has all 1 as mask
            bg_color = torch.rand(3, dtype=torch.float32, device=gaussians.device)
            # bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device) # NOTE: this is for shapenet cars, which do not have a gt mask
        else:
            bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)

        # use the other views for rendering and supervision
        results = self.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
        # st()
        pred_images = results['image'] # [B, V, C, output_size, output_size]
        pred_alphas = results['alpha'] # [B, V, 1, output_size, output_size]
        # st()
        
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
        

        if not torch.all(gt_masks==1):
            loss_mse = F.mse_loss(pred_images, gt_images) + F.mse_loss(pred_alphas, gt_masks) # NOTE: THIS IS disabled for srn_cars, which do not have a valid mask
        else:
            loss_mse = F.mse_loss(pred_images, gt_images)

        loss = loss + loss_mse
        
        
        # # print('train vids',[t.item() for t in data['vids']])
        # if (opt is not None) and (epoch % opt.save_train_pred)==0 and epoch > 0:
        # # if (opt is not None) and epoch:
        
        #     ### ----------- debug-------------
        #     gt_images_save = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
        #     gt_images_save = gt_images_save.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images_save.shape[1] * gt_images_save.shape[3], 3) # [B*output_size, V*output_size, 3]
        #     kiui.write_image(f'{opt.workspace}/train_gt_images_{epoch}_{i}.jpg', gt_images_save)

        #     # gt_masks_save = data['masks_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
        #     # gt_masks_save = gt_masks_save.transpose(0, 3, 1, 4, 2).reshape(-1, gt_masks_save.shape[1] * gt_masks_save.shape[3], 1) # [B*output_size, V*output_size, 3]
        #     # kiui.write_image(f'{opt.workspace}/train_gt_masks_{epoch}_{i}.jpg', gt_masks_save)

        #     pred_images_save = results['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
        #     pred_images_save = pred_images_save.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images_save.shape[1] * pred_images_save.shape[3], 3)
        #     kiui.write_image(f'{opt.workspace}/train_pred_images_{epoch}_{i}.jpg', pred_images_save)
        #     ### -------- debug [end]-------------

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