import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import kiui
from kiui.lpips import LPIPS
from diffusers import DiffusionPipeline, DDPMScheduler
from PIL import Image
import einops

from core.options import Options
from core.gs import GaussianRenderer

from ipdb import set_trace as st
import matplotlib.pyplot as plt
import os


from core.dataset_v5_marigold import ordered_attr_list, attr_map, sp_min_max_dict
from pytorch3d.transforms import quaternion_to_axis_angle, axis_angle_to_quaternion

def fuse_splatters(splatters):
    # fuse splatters
    B, V, C, H, W = splatters.shape

    x = splatters.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)
    return x

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

def to_rgb_image(maybe_rgba: Image.Image):
    if maybe_rgba.mode == 'RGB':
        return maybe_rgba
    elif maybe_rgba.mode == 'RGBA':
        rgba = maybe_rgba
        img = np.random.randint(127, 128, size=[rgba.size[1], rgba.size[0], 3], dtype=np.uint8)
        img = Image.fromarray(img, 'RGB')
        img.paste(rgba, mask=rgba.getchannel('A'))
        return img
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)
    

from itertools import chain
def optimizer_set_state(optimizer, state_dict):
    groups = optimizer.param_groups
    saved_groups = state_dict['param_groups']

    if len(groups) != len(saved_groups):
        raise ValueError("loaded state dict has a different number of "
                         "parameter groups")
    param_lens = (len(g['params']) for g in groups)
    saved_lens = (len(g['params']) for g in saved_groups)
    if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
        raise ValueError("loaded state dict contains a parameter group "
                         "that doesn't match the size of optimizer's group")

    # Update the state
    id_map = {old_id: p for old_id, p in
              zip(chain.from_iterable((g['params'] for g in saved_groups)),
                  chain.from_iterable((g['params'] for g in groups)))}

def attr_3channel_image_to_original_splatter_attr(attr_to_encode, mv_image):
        
        sp_image_o = 0.5 * (mv_image + 1) # [map to range [0,1]]
        
        if "scale" in attr_to_encode:
            # v2
            sp_min, sp_max = sp_min_max_dict["scale"]

            sp_image_o = sp_image_o.clip(0,1) 
            sp_image_o = sp_image_o * (sp_max - sp_min) + sp_min
            
            sp_image_o = torch.exp(sp_image_o)
            # print(f"Decoded attr [unscaled] {attr_to_encode}: min={sp_image_o.min()} max={sp_image_o.max()}")


        elif attr_to_encode in[ "pos"]:
            sp_min, sp_max = sp_min_max_dict[attr_to_encode]
            sp_image_o = sp_image_o * (sp_max - sp_min) + sp_min
            sp_image_o = torch.clamp(sp_image_o, min=sp_min, max=sp_max)
          

        if attr_to_encode == "rotation": 
            
            ag = einops.rearrange(sp_image_o, 'b c h w -> b h w c')
            quaternion = axis_angle_to_quaternion(ag)
            sp_image_o = einops.rearrange(quaternion, 'b h w c -> b c h w')
            # st()

        start_i, end_i = attr_map[attr_to_encode]
        if end_i - start_i == 1:
            sp_image_o = torch.mean(sp_image_o, dim=1, keepdim=True) # avg.
            
        return sp_image_o
    
def denormalize_and_activate(attr, mv_image):
    # mv_image: B C H W
    
    sp_image_o = 0.5 * (mv_image + 1) # [map to range [0,1]]
    
    if attr == "pos":
        sp_min, sp_max = sp_min_max_dict[attr]
        sp_image_o = sp_image_o * (sp_max - sp_min) + sp_min
        # sp_image_o = torch.clamp(sp_image_o, min=sp_min, max=sp_max)
    elif attr == "scale":
        sp_min, sp_max = sp_min_max_dict["scale"]
        # sp_image_o = sp_image_o.clip(0,1) 
        sp_image_o = sp_image_o * (sp_max - sp_min) + sp_min
        sp_image_o = torch.exp(sp_image_o)
    elif attr == "opacity":
        sp_image_o = sp_image_o.clip(0,1) 
        sp_image_o = torch.mean(sp_image_o, dim=1, keepdim=True) # avg.
    elif attr == "rotation": 
        # sp_image_o = sp_image_o.clip(0,1) 
        sp_min, sp_max = sp_min_max_dict["rotation"]
        sp_image_o = sp_image_o * (sp_max - sp_min) + sp_min
        ag = einops.rearrange(sp_image_o, 'b c h w -> b h w c')
        quaternion = axis_angle_to_quaternion(ag)
        sp_image_o = einops.rearrange(quaternion, 'b h w c -> b c h w')   
        
    return sp_image_o
    
    
class Interpolate(nn.Module):
    def __init__(self, size, mode='bilinear', align_corners=False):
        super(Interpolate, self).__init__()
        self.size = size
        self.mode = mode
        self.align_corners = align_corners if mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] else None

    def forward(self, x):
        x = F.interpolate(x, size=self.size, mode=self.mode, align_corners=self.align_corners)
        return x

        
class Zero123PlusGaussianMarigoldUnetCrossDomain(nn.Module):
    def __init__(
        self,
        opt: Options,
    ):
        super().__init__()
        self.opt = opt

        # Load zero123plus model
        import sys
        sys.path.append('./zero123plus')
    
        self.pipe = DiffusionPipeline.from_pretrained(
            opt.model_path,
            custom_pipeline=opt.custom_pipeline
        ).to('cuda')
        
        self.pipe.prepare() 
        self.vae = self.pipe.vae.requires_grad_(False).eval()
        self.unet = self.pipe.unet.requires_grad_(False).eval()
    
       
        self.pipe.scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config) # num_train_timesteps=1000
    
        # Gaussian Renderer
        self.gs = GaussianRenderer(opt)
        # activations...
        self.pos_act = lambda x: x.clamp(-1, 1)
        
        if opt.scale_bias_learnable:
            self.scale_bias = nn.Parameter(torch.tensor([opt.scale_act_bias]), requires_grad=False)
        else:
            self.scale_bias = opt.scale_act_bias
       
        if self.opt.scale_act == "biased_exp":
            max_scale = self.opt.scale_clamp_max # in torch.log scale
            min_scale = self.opt.scale_clamp_min
            # self.scale_act = lambda x: torch.exp(x + self.scale_bias)
            self.scale_act = lambda x: torch.exp(torch.clamp(x + self.scale_bias, max=max_scale, min=min_scale))
        elif self.opt.scale_act == "biased_softplus":
            max_scale = torch.exp(torch.tensor([self.opt.scale_clamp_max])).item() # in torch.log scale
            min_scale = torch.exp(torch.tensor([self.opt.scale_clamp_min])).item()
            # self.scale_act = lambda x: 0.1 * F.softplus(x + self.scale_bias)
            self.scale_act = lambda x: torch.clamp(0.1 * F.softplus(x + self.scale_bias), max=max_scale, min=min_scale)
        elif self.opt.scale_act == "softplus":
            # self.scale_act = lambda x: 0.1 * F.softplus(x)
            max_scale = torch.exp(torch.tensor([self.opt.scale_clamp_max])).item() # in torch.log scale
            min_scale = torch.exp(torch.tensor([self.opt.scale_clamp_min])).item()
            self.scale_act = lambda x: torch.clamp(0.1 * F.softplus(x), max=max_scale, min=min_scale)
        else: 
            raise ValueError ("Unsupported scale_act")
        
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = F.normalize
       
        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)
        

        # with open(f"{self.opt.workspace}/model_new.txt", "w") as f:
        #     print(self.unet, file=f)
      

    def get_alpha(self, scheduler, t, device):
        alphas_cumprod = scheduler.alphas_cumprod.to(
            device=device
        )
        alpha_t = alphas_cumprod[t] ** 0.5
        sigma_t = (1 - alphas_cumprod[t]) ** 0.5
        return alpha_t, sigma_t

    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict
 
    def build_optimizer(self, code_):
        optimizer_cfg = self.opt.optimizer.copy()
        optimizer_class = getattr(torch.optim, optimizer_cfg.pop('type'))
        if isinstance(code_, list):
            code_optimizer = [
                optimizer_class([code_single_], **optimizer_cfg)
                for code_single_ in code_]
        else:
            code_optimizer = optimizer_class([code_], **optimizer_cfg)
        return code_optimizer
    
    def build_splatter_optimizer(self, splatter_image):
        optimizer_cfg = self.opt.splatter_optimizer.copy()
        optimizer_class = getattr(torch.optim, optimizer_cfg.pop('type'))
        if isinstance(splatter_image, list):
            splatter_optimizer = [
                optimizer_class([splatter_single_], **optimizer_cfg)
                for splatter_single_ in splatter_image]
        else:
            splatter_optimizer = optimizer_class([splatter_image], **optimizer_cfg)
        return splatter_optimizer
    
    
    def forward(self, data, step_ratio=1, splatter_guidance=False, save_path=None, prefix=None):
        # Gaussian shape: (B*6, 14, H, W)

        results = {}
        loss = 0
      
        # encoder input: all the splatter attr pngs 
        images_all_attr_list = []
        for attr_to_encode in ordered_attr_list:
            sp_image = data[attr_to_encode]
            # print(f"[data]{attr_to_encode}: {sp_image.min(), sp_image.max()}")
            images_all_attr_list.append(sp_image)
        images_all_attr_batch = torch.stack(images_all_attr_list)
    
        A, B, _, _, _ = images_all_attr_batch.shape # [5, 1, 3, 384, 256]
        images_all_attr_batch = einops.rearrange(images_all_attr_batch, "A B C H W -> (B A) C H W")
    

        # save_path = f"{self.opt.workspace}/verify_bsz2"
        if save_path is not None:    
            images_to_save = images_all_attr_batch.detach().cpu().numpy() # [5, 3, output_size, output_size]
            images_to_save = (images_to_save + 1) * 0.5
            images_to_save = einops.rearrange(images_to_save, "a c (m h) (n w) -> (a h) (m n w) c", m=3, n=2)
            kiui.write_image(f'{save_path}/{prefix}images_all_attr_batch_to_encode.jpg', images_to_save)

        # do vae.encode
        sp_image_batch = scale_image(images_all_attr_batch)
        sp_image_batch = self.pipe.vae.encode(sp_image_batch).latent_dist.sample() * self.pipe.vae.config.scaling_factor
        latents_all_attr_encoded = scale_latents(sp_image_batch) # torch.Size([5, 4, 48, 32])
    
        if self.opt.custom_pipeline in ["./zero123plus/pipeline_v6_set.py", "./zero123plus/pipeline_v7_seq.py"]:
            gt_latents = latents_all_attr_encoded
        elif self.opt.cd_spatial_concat: # should use v2 pipeline
            gt_latents = einops.rearrange(latents_all_attr_encoded, "(B A) C (m H) (n W) -> B C (A H) (m n W)", B=data['cond'].shape[0], m=3, n=2)
        else:  
            raise NotImplementedError
        
        # prepare cond and t
        BA,C,H,W = gt_latents.shape # should be (B A) c h w
        assert (BA == B * A) or (self.opt.cd_spatial_concat)

        if self.opt.finetune_decoder:
            latents_all_attr_to_decode = gt_latents
            _rendering_w_t = 1
        elif self.opt.train_unet:
            print('Doing unet prediction')
            cond = data['cond'].unsqueeze(1).repeat(1,A,1,1,1).view(-1, *data['cond'].shape[1:]) 
            
            # unet 
            with torch.no_grad():
                text_embeddings, cross_attention_kwargs = self.pipe.prepare_conditions(cond, guidance_scale=1.0)
                cross_attention_kwargs_stu = cross_attention_kwargs        
        
            # same t for all domain
            # TODO: adapt this to batch_Size > 1 to run larger batch size
            t = torch.randint(0, self.pipe.scheduler.timesteps.max(), (B,), device=latents_all_attr_encoded.device)
            t = t.unsqueeze(1).repeat(1,A).view(-1)
            
            if self.opt.fixed_noise_level is not None:
                t = torch.ones_like(t).to(t.device) * self.opt.fixed_noise_level
                print(f"fixed noise level = {self.opt.fixed_noise_level}")
            # print("t=",t)
            
            noise = torch.randn_like(gt_latents, device=gt_latents.device)
            noisy_latents = self.pipe.scheduler.add_noise(gt_latents, noise, t)
            # print(noisy_latents.shape)
        
            domain_embeddings = torch.eye(5).to(noisy_latents.device)
            if self.opt.cd_spatial_concat:
                domain_embeddings = torch.sum(domain_embeddings, dim=0, keepdims=True) # feed all domains
            
            # add pos embedding on domain embedding
            domain_embeddings = torch.cat([
                    torch.sin(domain_embeddings),
                    torch.cos(domain_embeddings)
                ], dim=-1)
            
            # repeat for batch
            domain_embeddings = domain_embeddings.unsqueeze(0).repeat(B,1,1).view(-1, *domain_embeddings.shape[1:])

            # v-prediction with unet: (B A) 4 48 32
            v_pred = self.unet(noisy_latents, t, encoder_hidden_states=text_embeddings, cross_attention_kwargs=cross_attention_kwargs, class_labels=domain_embeddings).sample

            # get v_target
            alphas_cumprod = self.pipe.scheduler.alphas_cumprod.to(
                device=noisy_latents.device, dtype=noisy_latents.dtype
            )
            alpha_t = (alphas_cumprod[t] ** 0.5).view(-1, 1, 1, 1)
            sigma_t = ((1 - alphas_cumprod[t]) ** 0.5).view(-1, 1, 1, 1)
            v_target = alpha_t * noise - sigma_t * gt_latents # (B A) 4 48 32
        
            
            # calbculate loss
                # weight = alpha_t ** 2 / sigma_t ** 2 # SNR
            # reshape back to the batch to calculate loss?? loss is of shape [] without batch dim?
            loss_latent = F.mse_loss(v_pred, v_target)     
            results['loss_latent'] = loss_latent * self.opt.lambda_latent

            # calculate x0 from v_pred
            noise_pred = noisy_latents * sigma_t.view(-1, 1, 1, 1) + v_pred * alpha_t.view(-1, 1, 1, 1)
            x = (noisy_latents - noise_pred * sigma_t) / alpha_t
      
        
            if self.opt.cd_spatial_concat:
                latents_all_attr_to_decode = einops.rearrange(x, "B C (A H) (m n W) -> (B A) C (m H) (n W)", A=5, m=3, n=2)
            else:
                latents_all_attr_to_decode = x
            assert latents_all_attr_to_decode.shape == latents_all_attr_encoded.shape
            
            
            # Calculate rendering losses weights
            if self.opt.rendering_loss_use_weight_t:
                _alpha_t = alpha_t.flatten()[0]
                _rendering_w_t = _alpha_t ** 2 # TODOL make this to adapt B>1
                # print(f"_rendering_w_t of {t[0].item()}: ", _rendering_w_t)
            else:
                _rendering_w_t = 1
        else:
           raise NotImplementedError

        # vae.decode (batch process)
        latents_all_attr_to_decode = unscale_latents(latents_all_attr_to_decode)
        image_all_attr_to_decode = self.pipe.vae.decode(latents_all_attr_to_decode / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
        image_all_attr_to_decode = unscale_image(image_all_attr_to_decode) # (B A) C H W 
        # THIS IS IMPORTANT!! Otherwise the very small negative value will overflow when * 255 and converted to uint8
        image_all_attr_to_decode = image_all_attr_to_decode.clip(-1,1)

        # Reshape image_all_attr_to_decode from (B A) C H W -> A B C H W and enumerate on A dim
        image_all_attr_to_decode = einops.rearrange(image_all_attr_to_decode, "(B A) C H W -> A B C H W", B=B, A=A)
        
        # debug = False
        # if debug:
        #     image_all_attr_to_decode = einops.rearrange(images_all_attr_batch, "(B A) C H W -> A B C H W ", B=B, A=A)
        
        # decode latents into attrbutes again
        decoded_attr_list = []
        for i, _attr in enumerate(ordered_attr_list):
            batch_attr_image = image_all_attr_to_decode[i]
            # print(f"[vae.decode before]{_attr}: {batch_attr_image.min(), batch_attr_image.max()}")
            decoded_attr = denormalize_and_activate(_attr, batch_attr_image) # B C H W
            decoded_attr_list.append(decoded_attr)
            # print(f"[vae.decode after]{_attr}: {decoded_attr.min(), decoded_attr.max()}")
        

        if save_path is not None:
            # print('Saving to ', save_path)
            decoded_attr_3channel_image_batch = einops.rearrange(image_all_attr_to_decode, "A B C H W -> (B A) C H W ", B=B, A=A)
            images_to_save = decoded_attr_3channel_image_batch.to(torch.float32).detach().cpu().numpy() # [5, 3, output_size, output_size]
            images_to_save = (images_to_save + 1) * 0.5
            images_to_save = einops.rearrange(images_to_save, "a c (m h) (n w) -> (a h) (m n w) c", m=3, n=2)
            kiui.write_image(f'{save_path}/{prefix}images_all_attr_batch_decoded.jpg', images_to_save)

        splatter_mv = torch.cat(decoded_attr_list, dim=1) # [B, 14, 384, 256]
        # ## reshape 
        splatters_to_render = einops.rearrange(splatter_mv, 'b c (h2 h) (w2 w) -> b (h2 w2) c h w', h2=3, w2=2) # [1, 6, 14, 128, 128]
        results['splatters_from_code'] = splatters_to_render # [B, 6, 14, 256, 256]
        gaussians = fuse_splatters(splatters_to_render) # B, N, 14
        
        if self.training: # random bg for training
            bg_color = torch.rand(3, dtype=torch.float32, device=gaussians.device)
        else:
            bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device) * self.opt.bg # white bg

        #  render & calculate rendering loss
            
        ## ------- begin render ----------
        # use the other views for rendering and supervision
        if (self.opt.lambda_lpips + self.opt.lambda_rendering) > 0 and self.training:
            gs_results = self.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
        else:
            with torch.no_grad():
                gs_results = self.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
        
        pred_images = gs_results['image'] # [B, V, C, output_size, output_size]
        pred_alphas = gs_results['alpha'] # [B, V, 1, output_size, output_size]

        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas
        
        gt_images = data['images_output'] # [B, V, 3, output_size, output_size], ground-truth novel views
        gt_masks = data['masks_output'] # [B, V, 1, output_size, output_size], ground-truth masks
        
        # get gt_mask from gt gaussian rendering
        if self.opt.data_mode == "srn_cars":
            gt_masks = self.gs.render(fuse_splatters(data['splatters_output']), data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)['alpha']

        gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)
        
        # # save training pairs
        # pair_images = torch.cat([gt_images, pred_images]).detach().cpu().numpy() # [B, V, 3, output_size, output_size]
        # pair_images = pair_images.transpose(0, 3, 1, 4, 2).reshape(-1, pair_images.shape[1] * pair_images.shape[3], 3)
        # kiui.write_image(f'{self.opt.workspace}/train_gt_pred_images.jpg', pair_images)
        # st()
        ## ------- end render ----------

          
        if self.opt.lambda_rendering > 0:
            loss_mse_rendering = F.mse_loss(pred_images, gt_images) + F.mse_loss(pred_alphas, gt_masks)
            loss_mse_rendering *= _rendering_w_t
            results['loss_rendering'] = loss_mse_rendering
            # results['loss_rendering_rendering_w_t'] = loss_mse_rendering
            loss +=  self.opt.lambda_rendering * loss_mse_rendering
            if self.opt.verbose_main:
                print(f"loss rendering (with alpha):{loss_mse_rendering}")
        elif self.opt.lambda_alpha > 0:
            loss_mse_alpha = F.mse_loss(pred_alphas, gt_masks)
            loss_mse_alpha *= _rendering_w_t
            results['loss_alpha'] = loss_mse_alpha
            # results['loss_alpha_rendering_w_t'] = loss_mse_alpha
            loss += self.opt.lambda_alpha * loss_mse_alpha
            if self.opt.verbose_main:
                print(f"loss alpha:{loss_mse_alpha}")
            
        if self.opt.lambda_lpips > 0:
            loss_lpips = self.lpips_loss(
                # gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                # pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                # downsampled to at most 256 to reduce memory cost

                F.interpolate(gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
                F.interpolate(pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            ).mean()
            loss_lpips *= _rendering_w_t
            results['loss_lpips'] = loss_lpips
            # results['loss_lpips_weight-t'] = loss_lpips
            loss += self.opt.lambda_lpips * loss_lpips
            if self.opt.verbose_main:
                print(f"loss lpips:{loss_lpips}")
        
        if isinstance(loss, int):
            loss = torch.as_tensor(loss, device=psnr.device, dtype=psnr.dtype)
        results['loss'] = loss
        
        
        # Calculate metrics
        # TODO: add other metrics such as SSIM
        psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
        results['psnr'] = psnr.detach()
        
        
        return results
    