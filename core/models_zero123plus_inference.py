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
    def __init__(self, vae, opt):
        super(UNetDecoder, self).__init__()
        self.vae = vae
        self.decoder = vae.decoder
        
        self.decoder = self.decoder.requires_grad_(False).eval()
        # if (opt.decoder_mode in ["v1_fix_rgb", "v1_fix_rgb_remove_unscale"]) or (opt.inference_noise_level > 0):
        #     self.decoder = self.decoder.requires_grad_(False).eval()
        # else:
        #     self.decoder = self.decoder.requires_grad_(True).train()
        
        self.others = nn.Conv2d(128, 11, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        # if opt.verbose or opt.verbose_main:
        if True:
            print(f"opt.decoder_mode : {opt.decoder_mode}")    
            
            decoder_requires_grad = any(p.requires_grad for p in self.decoder.parameters()) ## check decoder requires grad
            print(f"UNet Decoder vae.decoder requires grad: {decoder_requires_grad}")

            others_requires_grad = any(p.requires_grad for p in self.others.parameters()) ## check decoder requires grad
            print(f"UNet Decoder others requires grad: {others_requires_grad}")
            # st()
    
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


class Zero123PlusGaussianInference(nn.Module):
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

        
        # Load the pipeline
        self.pipe_0123 = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.1", custom_pipeline=opt.custom_pipeline,
            torch_dtype=torch.float32
        ).to('cuda')
        self.pipe_0123.prepare()

        # self.pipe = DiffusionPipeline.from_pretrained(
        #     "sudo-ai/zero123plus-v1.1", custom_pipeline="sudo-ai/zero123plus-pipeline",
        #     torch_dtype=torch.float16
        # )
        
        
        self.pipe.prepare() 
        self.vae = self.pipe.vae.requires_grad_(False).eval()
        # self.vae.decoder.requires_grad_(True).train() #NOTE: this is done in the Unet Decoder

        ### insert the inference code
        ## scene: b0bce5ad99d84befaf9159681c551051

        # inference_in_init = 
        # if inference_in_init:
        #     print(f"Begin inference ...")
        #     guidance_scale = 4.0
        #     pipeline = self.pipe
        #     prompt_embeds, cak = pipeline.prepare_conditions(cond, guidance_scale=4.0)
        #     pipeline.scheduler.set_timesteps(1, device='cuda:0')
        #     timesteps = pipeline.scheduler.timesteps
        #     latents = torch.randn([1, pipeline.unet.config.in_channels, 120, 80], device='cuda:0', dtype=torch.float16)
        #     latents_init = latents.clone().detach()
            

        #     with torch.no_grad():
        #         for i, t in enumerate(timesteps):
        #             print(t)
        #             latent_model_input = torch.cat([latents] * 2)
        #             latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

        #             # predict the noise residual
        #             noise_pred = pipeline.unet(
        #                 latent_model_input,
        #                 t,
        #                 encoder_hidden_states=prompt_embeds,
        #                 cross_attention_kwargs=cak,
        #                 return_dict=False,
        #             )[0]

        #             # perform guidance
        #             if True:
        #                 noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        #                 noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        #             # noise_pred = predict_noise0_diffuser(pipeline.unet, latents, prompt_embeds, t, guidance_scale, cak, pipeline.scheduler)

        #             # compute the previous noisy sample x_t -> x_t-1
        #             latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        #         latents1 = unscale_latents(latents)
        #         image = pipeline.vae.decode(latents1 / pipeline.vae.config.scaling_factor, return_dict=False)[0]
        #         image = unscale_image(image)

        #         gt_images = image.float().detach().cpu().numpy() # [B, V, 3, output_size, output_size]
        #         # gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
        #         gt_images = gt_images.transpose(0, 2, 3, 1)
        #         gt_images = gt_images.reshape(-1, gt_images.shape[2], 3)
        #         kiui.write_image(f'inference_no_resume.jpg', gt_images)
        #         st()

        #         latents_init1 = unscale_latents(latents_init)
        #         image_init = pipeline.vae.decode(latents_init1 / pipeline.vae.config.scaling_factor, return_dict=False)[0]
        #         image_init = unscale_image(image_init)

        #         pred_images = image_init.float().detach().cpu().numpy() # [B, V, 3, output_size, output_size]
        #         # pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
        #         pred_images = pred_images.transpose(0, 2, 3, 1)
        #         kiui.write_image(f'init_image.jpg', pred_images)
                
        #         x = self.decode_latents(latents1)
                
        #         print(f"Finish inference ...")
        #         st()
        

        
        
        
        # ### --------- inference [end] ---------
        # if opt.train_unet:
        #     print("Unet is trainable")
        #     self.unet = self.pipe.unet.requires_grad_(True).train()
        # else:
        #     self.unet = self.pipe.unet.eval().requires_grad_(False)
        # self.pipe.scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config) # num_train_timesteps=1000

        # if opt.decoder_mode in ["v2_fix_rgb_more_conv"]:
        #     self.decoder = UNetDecoderV2(self.vae, opt)
        # elif opt.decoder_mode in ["v0_unfreeze_all", "v1_fix_rgb", "v1_fix_rgb_remove_unscale"]:
        #     self.decoder = UNetDecoder(self.vae, opt)
        # else:
        #     raise ValueError ("NOT a valid choice for decoder in Zero123PlusGaussian")
        self.decoder = UNetDecoder(self.vae, opt)
        self.decoder.requires_grad_(False).eval()
        # # with torch.no_grad():
        # #     cond = to_rgb_image(Image.open('/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/0a9b36d36e904aee8b51e978a7c0acfd/000.png'))
        # #     text_embeddings, cross_attention_kwargs = self.pipe.prepare_conditions(cond, guidance_scale=4.0)
        # #     cross_attention_kwargs_stu = cross_attention_kwargs

        # # add auxiliary layer for generating gaussian (14 channels)
        # # self.conv = nn.Conv2d(3, 14, kernel_size=3, stride=1, padding=1)

        # Gaussian Renderer
        self.gs = GaussianRenderer(opt)
        # activations...
        self.pos_act = lambda x: x.clamp(-1, 1)
        
       
        if opt.scale_bias_learnable:
            self.scale_bias = nn.Parameter(torch.tensor([opt.scale_act_bias]), requires_grad=True)
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
        if self.opt.decoder_mode == "v1_fix_rgb":
            self.rgb_act = lambda x: x
        elif self.opt.decoder_mode == "v1_fix_rgb_remove_unscale":
            self.rgb_act = lambda x: unscale_image(x)
        else:
            self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5 # NOTE: may use sigmoid if train again

        # # LPIPS loss
        # if self.opt.lambda_lpips > 0:
        #     self.lpips_loss = LPIPS(net='vgg')
        #     self.lpips_loss.requires_grad_(False)
    
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
            if self.opt.decoder_mode == "v1_fix_rgb_remove_unscale": 
                return image # do unscale for rgb only in rgb_act
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
    
    def forward_splatters_with_activation(self, images, cond, inference=False):
        B, V, C, H, W = images.shape
        # print(f"images.shape in forward+spaltter:{images.shape}") # SAME as the input_size
        with torch.no_grad():
            text_embeddings, cross_attention_kwargs = self.pipe.prepare_conditions(cond, guidance_scale=4.0)
            cross_attention_kwargs_stu = cross_attention_kwargs

        # make input 6 views into a 3x2 grid
        images = einops.rearrange(images, 'b (h2 w2) c h w -> b c (h2 h) (w2 w)', h2=3, w2=2) 

        # scale as in zero123plus
        latents = self.encode_image(images) # [1, 4, 48, 32]
        debug = False
        if self.opt.skip_predict_x0 and (not inference):
        # if self.opt.skip_predict_x0:
            x = self.decode_latents(latents)
        elif inference:
            import os
            import rembg
            guidance_scale = 4.0
            # img = to_rgb_image(Image.open(path))
            # img.save(os.path.join(output_path, f'{name}/cond.png'))
            # cond = [img]
            noise_level = 7
            pipeline = self.pipe_0123
            prompt_embeds, cak = pipeline.prepare_conditions(cond, guidance_scale=4.0)
            pipeline.scheduler.set_timesteps(noise_level, device='cuda:0')
            timesteps = pipeline.scheduler.timesteps
            # latents = torch.randn([1, pipeline.unet.config.in_channels, 120, 80], device='cuda:0', dtype=torch.float16)
            latents = torch.randn([1, pipeline.unet.config.in_channels, 48, 32], device='cuda:0', dtype=torch.float16)
            latents_init = latents.clone().detach()
            st()
            
            ######## ----- [BEGIN] ----- 
            with torch.no_grad():
                for i, t in enumerate(timesteps):
                    print(t)
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
                    # noise_pred = predict_noise0_diffuser(pipeline.unet, latents, prompt_embeds, t, guidance_scale, cak, pipeline.scheduler)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                latents1 = unscale_latents(latents)
                image = pipeline.vae.decode(latents1 / pipeline.vae.config.scaling_factor, return_dict=False)[0]
                image = unscale_image(image)

                latents_init1 = unscale_latents(latents_init)
                image_init = pipeline.vae.decode(latents_init1 / pipeline.vae.config.scaling_factor, return_dict=False)[0]
                image_init = unscale_image(image_init)
                
            mv_image = einops.rearrange((image[0].clip(-1,1)+1).cpu().numpy()*127.5, 'c (h2 h) (w2 w)-> (h2 w2) h w c', h2=3, w2=2).astype(np.uint8) 
            for i, image in enumerate(mv_image):
                image = rembg.remove(image).astype(np.float32) / 255.0
                if image.shape[-1] == 4:
                    image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
                im_path = os.path.join(self.opt.workspace, f'models_inference/{i:03d}.png')
                Image.fromarray((image * 255).astype(np.uint8)).save(im_path)
            print(f"Inference image saved to {im_path}")
            st()
            
        elif debug:   
            # print(f"Begin inference (image) ...")
            # guidance_scale = 4.0
            # pipeline = self.pipe_0123
            # prompt_embeds, cak = pipeline.prepare_conditions(cond, guidance_scale=4.0)
            # pipeline.scheduler.set_timesteps(noise_level, device='cuda:0')
            # timesteps = pipeline.scheduler.timesteps
            # latents = torch.randn([1, pipeline.unet.config.in_channels, 120, 80], device='cuda:0', dtype=torch.float16)
            # latents_init = latents.clone().detach()
            
            
            # print(f"End inference (image) ...")
            for _pi, pipeline in enumerate([self.pipe_0123, self.pipe]):
                print(f"Begin inference (for the {_pi}_th pipe) ...")
                guidance_scale = 4.0
                pipeline = self.pipe
                prompt_embeds, cak = pipeline.prepare_conditions(cond, guidance_scale=4.0)
                pipeline.scheduler.set_timesteps(100, device='cuda:0')
                timesteps = pipeline.scheduler.timesteps
                latents = torch.randn([1, pipeline.unet.config.in_channels, 120, 80], device='cuda:0', dtype=torch.float16)
                latents_init = latents.clone().detach()
                

                with torch.no_grad():
                    for i, t in enumerate(timesteps):
                        print(t)
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
                        # noise_pred = predict_noise0_diffuser(pipeline.unet, latents, prompt_embeds, t, guidance_scale, cak, pipeline.scheduler)

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                    latents1 = unscale_latents(latents)
                    image = pipeline.vae.decode(latents1 / pipeline.vae.config.scaling_factor, return_dict=False)[0]
                    image = unscale_image(image)

                    gt_images = image.float().detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    # gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                    gt_images = gt_images.transpose(0, 2, 3, 1)
                    gt_images = gt_images.reshape(-1, gt_images.shape[2], 3)
                    kiui.write_image(f'inference_no_resume.jpg', gt_images)
                    st()

                    latents_init1 = unscale_latents(latents_init)
                    image_init = pipeline.vae.decode(latents_init1 / pipeline.vae.config.scaling_factor, return_dict=False)[0]
                    image_init = unscale_image(image_init)

                    pred_images = image_init.float().detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    # pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    pred_images = pred_images.transpose(0, 2, 3, 1)
                    kiui.write_image(f'init_image.jpg', pred_images)

                    
                    # latents = self.encode_image(images) # [1, 4, 48, 32]
                    image = scale_image(image)
                    image = pipeline.vae.encode(image).latent_dist.sample() * pipeline.vae.config.scaling_factor
                    latents = scale_latents(image)
                    
                    self.decode_latents(latents)
                    latents = unscale_latents(latents)
                    latents = latents / self.vae.config.scaling_factor
                    # image = self.vae.decode(latents, return_dict=False)[0]
                    image = self.decoder(latents)
                    if self.opt.decoder_mode == "v1_fix_rgb_remove_unscale": 
                        return image # do unscale for rgb only in rgb_act
                    image = unscale_image(image)
                    
                    x = self.decode_latents(latents)
                    st()
                    print(f"Finish inference ...")
        else:
            noise_level = None
            t = torch.tensor([10] * B, device=latents.device)
            latents = self.pipe.scheduler.add_noise(latents, torch.randn_like(latents, device=latents.device), t)
            
            x = self.predict_x0(
                latents, text_embeddings, t=10, guidance_scale=1.0, 
                cross_attention_kwargs=cross_attention_kwargs, scheduler=self.pipe.scheduler, model='zero123plus')

            # t = torch.tensor([100] * B, device=latents.device)
            # latents = self.pipe.scheduler.add_noise(latents, torch.randn_like(latents, device=latents.device), t)
            
            # for step_t in self.scheduler.timesteps:
            #     print(f"step_t: {step_t}")
            #     latents = self.predict_x0(
            #         latents, text_embeddings, t=step_t, guidance_scale=1.0, 
            #         cross_attention_kwargs=cross_attention_kwargs, scheduler=self.pipe.scheduler, model='zero123plus')
            # x = latents
            
            # pipeline.scheduler.set_timesteps(75, device='cuda:0')
           
            # if inference:
            #     noise_level = self.opt.inference_noise_level
            # else:
            #     noise_level = 10

            # t = torch.tensor([noise_level] * B, device=latents.device)
            # latents = self.pipe.scheduler.add_noise(latents, torch.randn_like(latents, device=latents.device), t)
            
            # x = self.predict_x0(
            #     latents, text_embeddings, t=noise_level, guidance_scale=1.0, 
            #     cross_attention_kwargs=cross_attention_kwargs, scheduler=self.pipe.scheduler, model='zero123plus')
            
    
            print(f"pred x0: {x.shape}, latents:{latents.shape}, noise_level: {noise_level} (under inferce mode: {inference})")
            x = self.decode_latents(x) # (B, 14, H, W)
            # st()
                   
        
        # x = self.conv(x)

        x = x.permute(0, 2, 3, 1)
        
        pos = self.pos_act(x[..., :3]) # [B, N, 3]
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])
        
        if self.opt.verbose_main:
            print(f"self.scale bias: {self.scale_bias}")
            print(f"scale after clamp: max={torch.log(scale.max())}, min={torch.log(scale.min())}")
        
        device = x.device
        for attr in self.opt.normalize_scale_using_gt:
            st()
            if self.opt.verbose_main:
                print(f"Normalizing attr {attr} in forward splatttre")
            if attr == 'opacity':
                pred_attr_flatten = opacity
                gt_std = torch.tensor([[[3.2988]]], device=device)
                gt_mean = torch.tensor([[[-4.7325]]], device=device)
            elif attr == 'scale':
                pred_attr_flatten = scale
                gt_std = torch.tensor([[[1.0321],
                    [0.9340],
                    [1.0183]]], device=device)
                gt_mean = torch.tensor([[[-5.7224],
                    [-5.5628],
                    [-5.4192]]], device=device)

            else:
                raise ValueError ("This attribute is not supported for normalization")
            
            # gt_attr_flatten = torch.log(gt_attr_flatten).permute(0,2,1) # [B, C, L]
            b, H, W, c = pred_attr_flatten.shape
    
            pred_attr_flatten = einops.rearrange(pred_attr_flatten, 'b H W c -> b c (H W)') # # [B, C, L]
            pred_attr_flatten = torch.log(pred_attr_flatten)
            
            # TODO: Change the GT mean and var to be the one calculatd by the dataloader
        
            # # Assuming train_data has shape (B, C, L)
            # gt_mean = torch.mean(gt_attr_flatten, dim=(0, 2), keepdim=True) # [1, C, 1]
            # gt_std = torch.std(gt_attr_flatten, dim=(0, 2), keepdim=True)

            pred_mean = torch.mean(pred_attr_flatten, dim=(0, 2), keepdim=True) # [1, C, 1]
            pred_std = torch.std(pred_attr_flatten, dim=(0, 2), keepdim=True)

            # Normalize input_tensor to match the distribution of gt
        
            normalized_pred = (pred_attr_flatten - pred_mean) / (pred_std + 1e-5)  # Adding a small epsilon for numerical stability

            # If you want the normalized_input to have the same mean and std as gt_tensor
            pred_attr_flatten = normalized_pred * gt_std + gt_mean

            pred_attr_flatten = torch.exp(pred_attr_flatten) # because the norm is on the log scale
            pred_attr_flatten = einops.rearrange(pred_attr_flatten, 'b c (H W) -> b H W c', H=H, W=W) # [B, C, L]
            
            if attr == 'opacity':
                opacity = pred_attr_flatten 
            elif attr == 'scale':
                scale = pred_attr_flatten
            else:
                raise ValueError ("This attribute is not supported for normalization")

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
            # st()
            # print(f"dtype of splatter image: {pred_splatters}")
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
            ### check rgb output from zero123++
            # gt_splatters =  data['splatters_output'] # [1, 6, 14, 128, 128]
           
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
            
        
        if (self.opt.lambda_lpips + self.opt.lambda_rendering) > 0 or (not self.training):
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
        
        
        ### additional for inference
        if self.opt.inference_noise_level > 0:
            inference_splatters = self.forward_splatters_with_activation(images, cond, inference=True) # [B, N, 14] # (B, 6, 14, H, W)
            results['inference_splatters'] = inference_splatters # [1, 6, 14, 256, 256]
            if self.opt.verbose_main:
                print(f"model inference_splatters: {inference_splatters.shape}")
            
            gaussians = fuse_splatters(inference_splatters)
            bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device) * 0.5
            gs_results = self.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)

            inference_images = gs_results['image'] # [B, V, C, output_size, output_size]
            inference_alphas = gs_results['alpha'] # [B, V, 1, output_size, output_size]


            results['images_inference'] = inference_images
            results['alphas_inference'] = inference_alphas
            
            psnr = -10 * torch.log10(torch.mean((inference_images.detach() - gt_images) ** 2))
            results['psnr_inference'] = psnr
        

        return results
    