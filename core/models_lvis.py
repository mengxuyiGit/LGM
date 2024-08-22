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
        # self.rot_act = lambda x: torch.zeros_like(x)
        # self.rot_act = F.normalize
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5 # NOTE: may use sigmoid if train again

        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)
        


    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict


    # def prepare_default_rays(self, device, elevation=0):
        
    #     from kiui.cam import orbit_camera
    #     from core.utils import get_rays

    #     cam_poses = np.stack([
    #         orbit_camera(elevation, 0, radius=self.opt.cam_radius),
    #         orbit_camera(elevation, 90, radius=self.opt.cam_radius),
    #         orbit_camera(elevation, 180, radius=self.opt.cam_radius),
    #         orbit_camera(elevation, 270, radius=self.opt.cam_radius),
    #     ], axis=0) # [4, 4, 4]
    #     cam_poses = torch.from_numpy(cam_poses)

    #     rays_embeddings = []
    #     for i in range(cam_poses.shape[0]):
    #         rays_o, rays_d = get_rays(cam_poses[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
    #         rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
    #         rays_embeddings.append(rays_plucker)

    #         ## visualize rays for plotting figure
    #         # kiui.vis.plot_image(rays_d * 0.5 + 0.5, save=True)

    #     rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous().to(device) # [V, 6, h, w]
        
    #     return rays_embeddings
        

    def forward_gaussians(self, images):
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]

        B, V, C, H, W = images.shape
        images = images.view(B*V, C, H, W)

        x = self.unet(images) # [B*4, 14, h, w]
        x = self.conv(x) # [B*4, 14, h, w]

        x = x.reshape(B, 6, 14, self.opt.splat_size, self.opt.splat_size) # 4 -> 6 on the 2nd dim
        
        ## visualize multi-view gaussian features for plotting figure
        # tmp_alpha = self.opacity_act(x[0, :, 3:4])
        # tmp_img_rgb = self.rgb_act(x[0, :, 11:]) * tmp_alpha + (1 - tmp_alpha)
        # tmp_img_pos = self.pos_act(x[0, :, 0:3]) * 0.5 + 0.5
        # kiui.vis.plot_image(tmp_img_rgb, save=True)
        # kiui.vis.plot_image(tmp_img_pos, save=True)

        # print(f"gaussian resolution: {x.shape}")

        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)
       
        
        pos = self.pos_act(x[..., 0:3]) # [B, N, 3]
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        # print("rotations: ", rotation)
        rgbs = self.rgb_act(x[..., 11:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, N, 14]
        
        return gaussians

    
    def forward(self, data, step_ratio=1, iteration=-1):
        # data: output of the dataloader
        # return: loss

        results = {}
        loss = 0
      
        images = data['input'] # [B, 4, 9, h, W], input features
        
        # use the first view to predict gaussians
        gaussians = self.forward_gaussians(images) # [B, N, 14]

        #  # save gaussians
        # self.gs.save_ply(gaussians, 'gbuffer_medal.ply')
        # st()


        # random bg for training
        if self.training:
            bg_color = torch.rand(3, dtype=torch.float32, device=gaussians.device)
        else:
            bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)

        # use the other views for rendering and supervision
        results = self.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
        pred_images = results['image'] # [B, V, C, output_size, output_size]
        pred_alphas = results['alpha'] # [B, V, 1, output_size, output_size]

        results['gaussians'] = gaussians
        
        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas

        gt_images = data['images_output'] # [B, V, 3, output_size, output_size], ground-truth novel views
        gt_masks = data['masks_output'] # [B, V, 1, output_size, output_size], ground-truth masks

        gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)

        loss_mse = F.mse_loss(pred_images, gt_images) + F.mse_loss(pred_alphas, gt_masks)
        results['loss_mse'] = loss_mse
        loss = loss + loss_mse

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
            
        
        ### 2dgs regularizations
        lambda_normal = self.opt.lambda_normal if iteration > self.opt.normal_depth_begin_iter else 0.0 # instantmesh also introduced normal loss at the 2nd stage
        lambda_depth = self.opt.lambda_depth if iteration > self.opt.normal_depth_begin_iter else 0.0
        lambda_normal_err = self.opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = self.opt.lambda_dist if iteration > 3000 else 0.0
        # print(f"Iteration: {iteration}, lambda_normal: {lambda_normal}, lambda_normal_err: {lambda_normal_err} lambda_dist: {lambda_dist}")

        rend_dist = results["rend_dist"]
        rend_normal  = results['rend_normal']
        surf_normal = results['surf_normal']


        # detach = True
        # if detach:
        #     normal_error = (1 - (data['normals_output'] * surf_normal).sum(dim=0))[None]
        #     print('normal_error detached')
        # else:
        #     normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        # normal_err = lambda_normal_err * (normal_error).mean()
        normal_err = torch.tensor([0.0], device=gaussians.device)
        
        dist_loss = lambda_dist * (rend_dist).mean()

        # loss
        loss = loss + dist_loss + normal_err
        results['dist_loss'] = normal_err
        results['normal_err'] = normal_err
        
        # TODO
        # 1. add normal loss wih gt
       
        render_normals = rend_normal
        target_normals =  data['normals_output'] # [B, V, 3, output_size, output_size]
        similarity = (render_normals * target_normals).sum(dim=-3).abs() # both are within [-1,1]
        normal_mask = gt_masks.squeeze(-3)
        loss_normal = 1 - similarity[normal_mask>0].mean()
        loss_normal = lambda_normal * loss_normal
        
        loss += loss_normal
        results['normal_loss'] = loss_normal
        
        
        # # 2. add depth loss with gt
        # render_depths = results['surf_depth']
        # target_depths = data['depths_output'] # [B, V, 1, output_size, output_size]
        # target_alphas = gt_masks
        # loss_depth = lambda_depth * F.l1_loss(render_depths[target_alphas>0], target_depths[target_alphas>0])

        # loss += loss_depth
        # results['depth_loss'] = loss_depth
        
        # # 3. add larger alpha loss, which to ensure the normal will add up to 1
        # print('alpha range: ', pred_alphas.min(), pred_alphas.max())
       
        
        if iteration % 500 == 0:
            print(f"Iteration: {iteration}, lambda_normal: {lambda_normal}, lambda_depth: {lambda_depth}, lambda_normal_err: {lambda_normal_err} lambda_dist: {lambda_dist}")
            # print(f"Iteration: {iteration}, normal_loss: {loss_normal}, depth_loss: {loss_depth}, dist_loss: {dist_loss}, normal_err: {normal_err}")
            print(f"Iteration: {iteration}, normal_loss: {loss_normal}, dist_loss: {dist_loss}, normal_err: {normal_err}")
    
        
        results['loss'] = loss

        # metric
        with torch.no_grad():
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
            results['psnr'] = psnr

        return results