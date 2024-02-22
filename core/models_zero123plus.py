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



gt_attr_keys = ['pos', 'opacity', 'scale', 'rotation', 'rgbs']
start_indices = [0, 3, 4, 7, 11]
end_indices = [3, 4, 7, 11, 14]

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

def predict_noise0_diffuser(unet, noisy_latents, text_embeddings, t, 
    guidance_scale=1.0, cross_attention_kwargs={}, scheduler=None, lora_v=False, model='sd'):
    batch_size = noisy_latents.shape[0]
    latent_model_input = torch.cat([noisy_latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    if type(t) == int:
        t = torch.tensor([t] * batch_size, device=noisy_latents.device)
    # https://github.com/threestudio-project/threestudio/blob/77de7d75c34e29a492f2dda498c65d2fd4a767ff/threestudio/models/guidance/stable_diffusion_vsd_guidance.py#L512
    alphas_cumprod = scheduler.alphas_cumprod.to(
        device=noisy_latents.device, dtype=noisy_latents.dtype
    )
    alpha_t = alphas_cumprod[t] ** 0.5
    sigma_t = (1 - alphas_cumprod[t]) ** 0.5
    if guidance_scale == 1.:
        cak = {}
        if cross_attention_kwargs is not None:
            for key in cross_attention_kwargs:
                if isinstance(cross_attention_kwargs[key], torch.Tensor):
                    cak[key] = cross_attention_kwargs[key][batch_size:]
                else:
                    cak[key] = cross_attention_kwargs[key]
        noise_pred = unet(noisy_latents, t, encoder_hidden_states=text_embeddings[batch_size:], cross_attention_kwargs=cak).sample
        if lora_v or scheduler.config.prediction_type == 'v_prediction':
            # assume the output of unet is v-pred, convert to noise-pred now
            noise_pred = noisy_latents * sigma_t.view(-1, 1, 1, 1) + noise_pred * alpha_t.view(-1, 1, 1, 1)
        if model == 'unet_if':
            noise_pred, _ = noise_pred.split(3, dim=1)
    else:
        t = torch.cat([t] * 2)
        # predict the noise residual
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings, cross_attention_kwargs=cross_attention_kwargs).sample
        if lora_v or scheduler.config.prediction_type == 'v_prediction':
            # assume the output of unet is v-pred, convert to noise-pred now
            noise_pred = latent_model_input * torch.cat([sigma_t] * 2, dim=0).view(-1, 1, 1, 1) + noise_pred * torch.cat([alpha_t] * 2, dim=0).view(-1, 1, 1, 1)
        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        if model == 'unet_if':
            noise_pred_text, predicted_variance = noise_pred_text.split(3, dim=1)
            noise_pred_uncond, _ = noise_pred_uncond.split(3, dim=1)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred

# From: https://github.com/huggingface/diffusers/blob/v0.26.3/src/diffusers/models/autoencoders/autoencoder_kl.py#L35
class UNetDecoder(nn.Module):
    def __init__(self, vae):
        super(UNetDecoder, self).__init__()
        self.vae = vae
        self.decoder = vae.decoder
        self.others = nn.Conv2d(128, 11, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
    def forward(self, z):
        sample = self.vae.post_quant_conv(z)
        latent_embeds = None
        sample = self.decoder.conv_in(sample)
        upscale_dtype = next(iter(self.decoder.up_blocks.parameters())).dtype
        sample = self.decoder.mid_block(sample, latent_embeds)
        sample = sample.to(upscale_dtype)
        # up
        for up_block in self.decoder.up_blocks:
            sample = up_block(sample, latent_embeds)

        sample = self.decoder.conv_norm_out(sample)
        sample = self.decoder.conv_act(sample)
        rgb = self.decoder.conv_out(sample)
        others = self.others(sample)
        return torch.cat([others, rgb], dim=1)
        # return rgb


class Zero123PlusGaussian(nn.Module):
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
        
        # # Load the pipeline
        # self.pipe = DiffusionPipeline.from_pretrained(
        #     "sudo-ai/zero123plus-v1.1", custom_pipeline="sudo-ai/zero123plus-pipeline",
        #     torch_dtype=torch.float16
        # )

        self.pipe.prepare() 
        self.vae = self.pipe.vae.requires_grad_(False).eval()
        self.vae.decoder.requires_grad_(True).train()

        if opt.train_unet:
            print("Unet is trainable")
            self.unet = self.pipe.unet.requires_grad_(True).train()
        else:
            self.unet = self.pipe.unet.eval().requires_grad_(False)
        self.pipe.scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config) # num_train_timesteps=1000
        self.decoder = UNetDecoder(self.vae)

        # with torch.no_grad():
        #     cond = to_rgb_image(Image.open('/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/0a9b36d36e904aee8b51e978a7c0acfd/000.png'))
        #     text_embeddings, cross_attention_kwargs = self.pipe.prepare_conditions(cond, guidance_scale=4.0)
        #     cross_attention_kwargs_stu = cross_attention_kwargs

        # add auxiliary layer for generating gaussian (14 channels)
        # self.conv = nn.Conv2d(3, 14, kernel_size=3, stride=1, padding=1)

        # Gaussian Renderer
        self.gs = GaussianRenderer(opt)
        # activations...
        self.pos_act = lambda x: x.clamp(-1, 1)
        if self.opt.scale_act == "biased_exp":
            bias = opt.scale_act_bias
            self.scale_act = lambda x: torch.exp(x + bias)
        elif self.opt.scale_act == "biased_softplus":
            bias = opt.scale_act_bias
            self.scale_act = lambda x: 0.1 * F.softplus(x + bias)
        elif self.opt.scale_act == "softplus":
            self.scale_act = lambda x: 0.1 * F.softplus(x)
        else: 
            raise ValueError ("Unsupported scale_act")
        
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = F.normalize
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5 # NOTE: may use sigmoid if train again

        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)
    
    def encode_image(self, image, is_zero123plus=True):
        # st() # image: torch.Size([1, 3, 768, 512])

        if is_zero123plus:
            image = scale_image(image)
            image = self.vae.encode(image).latent_dist.sample() * self.vae.config.scaling_factor
            image = scale_latents(image)
        else:
            image = self.vae.encode(image, return_dict=False)[0] * self.vae.config.scaling_factor
        return image

    def decode_latents(self, latents, is_zero123plus=True):
        if is_zero123plus:
            latents = unscale_latents(latents)
            latents = latents / self.vae.config.scaling_factor
            # image = self.vae.decode(latents, return_dict=False)[0]
            image = self.decoder(latents)
            image = unscale_image(image)
        else:
            image = self.vae.decode(latents, return_dict=False)[0]
        return image

    def get_alpha(self, scheduler, t, device):
        alphas_cumprod = scheduler.alphas_cumprod.to(
            device=device
        )
        alpha_t = alphas_cumprod[t] ** 0.5
        sigma_t = (1 - alphas_cumprod[t]) ** 0.5
        return alpha_t, sigma_t
    
    def predict_x0(self, noisy_latents, text_embeddings, t, 
            guidance_scale=1.0, cross_attention_kwargs={}, 
            scheduler=None, lora_v=False, model='sd'):
        alpha_t, sigma_t = self.get_alpha(scheduler, t, noisy_latents.device)
        noise_pred = predict_noise0_diffuser(
            self.unet, noisy_latents, text_embeddings, t=t,
            guidance_scale=guidance_scale, cross_attention_kwargs=cross_attention_kwargs, 
            scheduler=scheduler, model=model
        )
        return (noisy_latents - noise_pred * sigma_t) / alpha_t

    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict

    def forward_gaussians(self, images, cond):
        B, V, C, H, W = images.shape
        with torch.no_grad():
            text_embeddings, cross_attention_kwargs = self.pipe.prepare_conditions(cond, guidance_scale=4.0)
            cross_attention_kwargs_stu = cross_attention_kwargs

        # make input 6 views into a 3x2 grid
        images = einops.rearrange(images, 'b (h2 w2) c h w -> b c (h2 h) (w2 w)', h2=3, w2=2) 

        # scale as in zero123plus
        latents = self.encode_image(images)
        t = torch.tensor([10] * B, device=latents.device)
        latents = self.pipe.scheduler.add_noise(latents, torch.randn_like(latents, device=latents.device), t)
        x = self.predict_x0(
            latents, text_embeddings, t=10, guidance_scale=1.0, 
            cross_attention_kwargs=cross_attention_kwargs, scheduler=self.pipe.scheduler, model='zero123plus')
        # x = torch.randn([B, 4, 96, 64], device=images.device)
        x = self.decode_latents(x) # (B, 14, H, W)
        # x = self.conv(x)

        x = einops.rearrange(x, 'b c (h2 h) (w2 w) -> b (h2 w2) c h w', h2=3, w2=2) # (B, 6, 14, H, W)
        x = x.reshape(B*6, -1, H, W)

        x = x.reshape(B, V, 14, 256, 256)
        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)
        
        pos = self.pos_act(x[..., :3]) # [B, N, 3]
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = x[..., 11:] # FIXME: original activation removed

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, N, 14]
        return gaussians
    
    def forward_splatters(self, images, cond):
        B, V, C, H, W = images.shape
        # print(f"images.shape in forward+spaltter:{images.shape}") # SAME as the input_size
        with torch.no_grad():
            text_embeddings, cross_attention_kwargs = self.pipe.prepare_conditions(cond, guidance_scale=4.0)
            cross_attention_kwargs_stu = cross_attention_kwargs

        # make input 6 views into a 3x2 grid
        images = einops.rearrange(images, 'b (h2 w2) c h w -> b c (h2 h) (w2 w)', h2=3, w2=2) 

        # scale as in zero123plus
        latents = self.encode_image(images) # [1, 4, 48, 32]
        
        t = torch.tensor([10] * B, device=latents.device)
        latents = self.pipe.scheduler.add_noise(latents, torch.randn_like(latents, device=latents.device), t)
        x = self.predict_x0(
            latents, text_embeddings, t=10, guidance_scale=1.0, 
            cross_attention_kwargs=cross_attention_kwargs, scheduler=self.pipe.scheduler, model='zero123plus')
        # x = torch.randn([B, 4, 96, 64], device=images.device)
        x = self.decode_latents(x) # (B, 14, H, W)
        # x = self.conv(x)

        x = einops.rearrange(x, 'b c (h2 h) (w2 w) -> b (h2 w2) c h w', h2=3, w2=2) # (B, 6, 14, H, W)
        splatters = x
        

        return splatters
        
        x = x.reshape(B*6, -1, H, W)

        x = x.reshape(B, V, 14, H, W)
        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)
        
        pos = self.pos_act(x[..., :3]) # [B, N, 3]
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = x[..., 11:] # FIXME: original activation removed

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, N, 14]
        return gaussians, splatters
    
    def forward_splatters_with_activation(self, images, cond):
        B, V, C, H, W = images.shape
        # print(f"images.shape in forward+spaltter:{images.shape}") # SAME as the input_size
        with torch.no_grad():
            text_embeddings, cross_attention_kwargs = self.pipe.prepare_conditions(cond, guidance_scale=4.0)
            cross_attention_kwargs_stu = cross_attention_kwargs

        # make input 6 views into a 3x2 grid
        images = einops.rearrange(images, 'b (h2 w2) c h w -> b c (h2 h) (w2 w)', h2=3, w2=2) 

        # scale as in zero123plus
        latents = self.encode_image(images) # [1, 4, 48, 32]
        
        if self.opt.skip_predict_x0:
            x = self.decode_latents(latents)
        else:
            t = torch.tensor([10] * B, device=latents.device)
            latents = self.pipe.scheduler.add_noise(latents, torch.randn_like(latents, device=latents.device), t)
            x = self.predict_x0(
                latents, text_embeddings, t=10, guidance_scale=1.0, 
                cross_attention_kwargs=cross_attention_kwargs, scheduler=self.pipe.scheduler, model='zero123plus')
            # x = torch.randn([B, 4, 96, 64], device=images.device)
            print(f"pred x0: {x.shape}, latents:{latents.shape}")
            x = self.decode_latents(x) # (B, 14, H, W)

            st()
        
        # x = self.conv(x)

        x = x.permute(0, 2, 3, 1)
        
        pos = self.pos_act(x[..., :3]) # [B, N, 3]
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])

        ## TODO: clamp
        # print(f"pred scale max:{scale.max()} scale min: {scale.min()}")
        # clamp the min and max scale to avoid oom
        # st()
        # print("clamp the scale to 1e-4")
        # scale = torch.clamp(scale, max=1e-4)
        
        rotation = self.rot_act(x[..., 7:11])
        # rgbs = x[..., 11:] # FIXME: original activation removed
        rgbs = self.rgb_act(x[..., 11:])

        splatters = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, N, 14]
        
        splatters = einops.rearrange(splatters, 'b (h2 h) (w2 w) c -> b (h2 w2) c h w', h2=3, w2=2) # (B, 6, 14, H, W)
        return splatters
    
    
    
    def gs_weighted_mse_loss(self, pred_splatters, gt_splatters):
        # ORIGINAL :loss_mse = F.mse_loss(pred_splatters, gt_splatters)
        ## TODO: make it even smarter: dynamically adjusting the weights for each attributes!!
        ## 
        # ipdb> abs(gt_splatters_flat).mean(dim=-2)
        # tensor([[0.1491, 0.2690, 0.1600, 
        #           0.1925, 
        #           0.0054, 0.0062, 0.0076, 
        #           0.0023, 0.0023, 0.0023, 0.0024, 
        #           0.4818, 0.4843, 0.4778]], device='cuda:0')
        
     
        gt_splatters = einops.rearrange(gt_splatters, 'b v c h w -> b (v h w) c')
        pred_splatters = einops.rearrange(pred_splatters, 'b v c h w -> b (v h w) c')
        
        ## rotation is not that important. Rgb scaling matters
        attr_weight_dict = {
            'pos':1, 'opacity':1e-3, 'scale':1e-3, 'rotation':1, 'rgbs':1
        }
        
        ## the common start indices and keys are global variables
       
        attr_weighted_loss_dict = {}
        
        total_loss_weighted = 0
        for key, si, ei in zip (gt_attr_keys, start_indices, end_indices):
            
            attr_weight = attr_weight_dict[key]
            
            gt_attr = gt_splatters[..., si:ei]
            pred_attr = pred_splatters[..., si:ei]
            
            if key in self.opt.attr_use_logrithm_loss:
                # print(f"apply log loss to {key}")
                # mse_before_log = F.mse_loss(pred_attr, gt_attr)
                # print(f"loss before apply logrithm{mse_before_log}")
                # attr_weighted_loss_dict[f"{key}_before_log"] = mse_before_log
                gt_attr = torch.log(gt_attr)
                pred_attr = torch.log(pred_attr)
                # print(f"loss after apply logrithm{F.mse_loss(pred_attr, gt_attr)}")
                # st()
            
            attr_weighted_loss = F.mse_loss(pred_attr, gt_attr) * attr_weight
            attr_weighted_loss_dict.update({key: attr_weighted_loss})
            total_loss_weighted += attr_weighted_loss
           
            if self.opt.verbose_main:
                print("--loss-", key, attr_weight, attr_weighted_loss)
        
        attr_weighted_loss_dict.update({'total': total_loss_weighted})

        return attr_weighted_loss_dict
        
        
    def forward(self, data, step_ratio=1):
        # Gaussian shape: (B*6, 14, H, W)
        
        results = {}
        loss = 0
       
        images = data['input'] # [B, 6, 9, h, W], input features: splatter images
        cond = data['cond'] # [B, H, W, 3], condition image
        
        # use the first view to predict gaussians
       
        pred_splatters = self.forward_splatters_with_activation(images, cond) # [B, N, 14] # (B, 6, 14, H, W)
        results['splatters_pred'] = pred_splatters # [1, 6, 14, 256, 256]
        if self.opt.verbose_main:
            print(f"model splatters_pred: {pred_splatters.shape}")
        
        
        if self.opt.lambda_splatter > 0:
            gt_splatters =  data['splatters_output'] # [1, 6, 14, 128, 128]
            # if self.opt.discard_small_opacities: # only for gt debug
            #     opacity = gt_splatters[:,:,3:4]
            #     mask = opacity.squeeze(-1) >= 0.005
            #     mask = mask.repeat(1,1,14,1,1)
            #     print(mask.shape)
            #     st()
                
            # loss_mse_unweighted = F.mse_loss(pred_splatters, gt_splatters)
            
            gs_loss_mse_dict = self.gs_weighted_mse_loss(pred_splatters, gt_splatters)
            loss_mse = gs_loss_mse_dict['total']
            # st()
            results['loss_splatter'] = loss_mse
            loss = loss + self.opt.lambda_splatter * loss_mse
            # also log the losses for each attributes
            results['gs_loss_mse_dict'] = gs_loss_mse_dict
            


        ## ------- splatter -> gaussian ------- 
    
        gaussians = fuse_splatters(pred_splatters)
                   
        
        if self.opt.render_gt_splatter:
            print("Render GT splatter --> load splatters then fuse")
            gaussians = fuse_splatters(data['splatters_output'])
            # print("directly load fused gaussian ply")
            # gaussians = self.gs.load_ply('/home/xuyimeng/Repo/LGM/data/splatter_gt_full/00000-hydrant-eval_pred_gs_6100_0/fused.ply').to(pred_splatters.device)
            # gaussians = gaussians.unsqueeze(0)
            
            ## TODO: clamp reference
            # scale = gaussians[...,4:7]
            # print(f"gt splatter scale max:{scale.max()} scale min: {scale.min()}")
            # # clamp the min and max scale to avoid oom
            # st()
            
            if self.opt.perturb_rot_scaling:
                # scaling: 4-7
                # gaussians[...,4:7] = 0.006 * torch.rand_like(gaussians[...,4:7])
                # rot: 7-11
                # gaussians[...,7:11] = F.normalize(-1 + 2 * torch.zeros_like(gaussians[...,7:11]))
                # print("small scale")
                pass
                
            
            if self.opt.discard_small_opacities: # only for gt debug
                opacity = gaussians[...,3:4]
                mask = opacity.squeeze(-1) >= 0.005
                gaussians = gaussians[mask].unsqueeze(0)
       
        elif len(self.opt.gt_replace_pred) > 0: 
            ### this will work
            # gaussians = fuse_splatters(data['splatters_output'])
            # print(f"replace --> assign")
            
            ### this will NOT work!!!!?
            gt_gaussians = fuse_splatters(data['splatters_output'])
            st()
            gaussians[..., :] = gt_gaussians[..., :]
            # del gt_gaussians
            print(f"replace --> slicing")
            
            # attr_map = {key: (si, ei) for key, si, ei in zip (gt_attr_keys, start_indices, end_indices)}
            # for attr_to_replace in self.opt.gt_replace_pred:
            #     start_i, end_i = attr_map[attr_to_replace]          
            #     print(f"replace pred__['{attr_to_replace}']__ with GT splatter --> load splatters then fuse") 
            #     gaussians[..., start_i:end_i] = gt_gaussians[..., start_i:end_i]
            
            # test_rand_seed = torch.randint(0, 100, (3,))
            # print(f"test_rand_seed: {test_rand_seed}")
            # st()
           
        
        # if (self.opt.lambda_lpips + self.opt.lambda_rendering) > 0 or (not self.training):
        if True:
            if self.opt.verbose_main:
                print(f"Render when self.training = {self.training}")
                
            ## ------- begin render ----------
            # random bg for training
            if self.training:
                bg_color = torch.rand(3, dtype=torch.float32, device=gaussians.device)
            else:
                bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device) * 0.5

            # use the other views for rendering and supervision
            gs_results = self.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
            pred_images = gs_results['image'] # [B, V, C, output_size, output_size]
            pred_alphas = gs_results['alpha'] # [B, V, 1, output_size, output_size]

            # FIXME: duplicate items with different keys? (in dict:results)
            results['images_pred'] = pred_images
            results['alphas_pred'] = pred_alphas
            
            gt_images = data['images_output'] # [B, V, 3, output_size, output_size], ground-truth novel views
            gt_masks = data['masks_output'] # [B, V, 1, output_size, output_size], ground-truth masks

            gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)

            ## ------- end render ----------
            
            # ----- FIXME: calculate psnr shoud be at the end -----
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
            results['psnr'] = psnr
        
       
            
            
        if self.opt.lambda_rendering > 0:
            loss_mse_rendering = F.mse_loss(pred_images, gt_images) + F.mse_loss(pred_alphas, gt_masks)
            results['loss_rendering'] = loss_mse_rendering
            loss = loss + self.opt.lambda_rendering * loss_mse_rendering
            if self.opt.verbose_main:
                print(f"loss rendering (with alpha):{loss_mse_rendering}")
        elif self.opt.lambda_alpha > 0:
            loss_mse_alpha = F.mse_loss(pred_alphas, gt_masks)
            results['loss_alpha'] = loss_mse_alpha
            loss = loss + self.opt.lambda_alpha * loss_mse_alpha
            if self.opt.verbose_main:
                print(f"loss alpha:{loss_mse_alpha}")
            
    

        ## FIXME: it does not make sense to apply lpips on splatter, right?
        if self.opt.lambda_lpips > 0:
            loss_lpips = self.lpips_loss(
                # gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                # pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                # downsampled to at most 256 to reduce memory cost
                
                # FIXME: change the dim to 14 for splatter imaegs
                F.interpolate(gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
                F.interpolate(pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            ).mean()
            results['loss_lpips'] = loss_lpips
            loss = loss + self.opt.lambda_lpips * loss_lpips
            if self.opt.verbose_main:
                print(f"loss lpips:{loss_lpips}")
            
        

        # ----- rendering [end] -----
        psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
        results['psnr'] = psnr
            
        
        results['loss'] = loss

        return results
    