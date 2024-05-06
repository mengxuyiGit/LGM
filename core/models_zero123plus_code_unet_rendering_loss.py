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
        st()
        noise_pred = unet(noisy_latents, t, encoder_hidden_states=text_embeddings[batch_size:], cross_attention_kwargs=cak).sample
        if lora_v or scheduler.config.prediction_type == 'v_prediction':
            # assume the output of unet is v-pred, convert to noise-pred now
            noise_pred = noisy_latents * sigma_t.view(-1, 1, 1, 1) + noise_pred * alpha_t.view(-1, 1, 1, 1)
        if model == 'unet_if':
            st()
            noise_pred, _ = noise_pred.split(3, dim=1)
    else:
        t = torch.cat([t] * 2)
        st()
        # predict the noise residual
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings, cross_attention_kwargs=cross_attention_kwargs).sample
        if lora_v or scheduler.config.prediction_type == 'v_prediction':
            # assume the output of unet is v-pred, convert to noise-pred now
            st()
            noise_pred = latent_model_input * torch.cat([sigma_t] * 2, dim=0).view(-1, 1, 1, 1) + noise_pred * torch.cat([alpha_t] * 2, dim=0).view(-1, 1, 1, 1)
        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        if model == 'unet_if':
            noise_pred_text, predicted_variance = noise_pred_text.split(3, dim=1)
            noise_pred_uncond, _ = noise_pred_uncond.split(3, dim=1)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred

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

class DownsampleModuleOnlyConvAvgPool(nn.Module):
    def __init__(self, input_channels, output_channels, input_resolution, output_resolution, use_activation_at_downsample=True):
        super(DownsampleModuleOnlyConvAvgPool, self).__init__()

        # Calculate the number of downsampling operations needed

        layers = []
        current_channels = input_channels
        
        layers.append(nn.Conv2d(current_channels, output_channels, kernel_size=3, stride=2, padding=1))
           
        layers.append(nn.AdaptiveAvgPool2d((3*output_resolution, 2*output_resolution)))

        self.downsample_layers = nn.Sequential(*layers)
    
    
    def forward(self, x):
        return self.downsample_layers(x)
        # for ly in self.downsample_layers:
        #     print(f"---{ly._get_name()}---")
        #     print(f"input size:{x.shape}")
        #     x = ly(x)
        #     print(f"output size:{x.shape}")
        # st()
        # return x
        
        
        
class DownsampleModule(nn.Module):
    def __init__(self, input_channels, output_channels, input_resolution, output_resolution, use_activation_at_downsample=True):
        super(DownsampleModule, self).__init__()

        # Calculate the number of downsampling operations needed
        num_downsamples = int(torch.log2(torch.tensor(input_resolution / output_resolution)))

        layers = []
        current_channels = input_channels
        num_groups = input_channels
        
        # for _ in range(num_downsamples - 1):  # We leave one less downsample to use adaptive pooling later
        for _ in range(num_downsamples):
            layers.append(nn.Conv2d(current_channels, current_channels * 2, kernel_size=3, stride=2, padding=1))
            current_channels *= 2
            # layers.append(nn.BatchNorm2d(current_channels * 2))
            # layers.append(nn.ReLU(inplace=True))
            layers.append(nn.GroupNorm(num_groups, current_channels, eps=1e-06, affine=True))
            layers.append(nn.SiLU())

        layers.append(nn.Conv2d(current_channels, output_channels, kernel_size=3, stride=1, padding=1))
      
        if use_activation_at_downsample:
            # layers.append(nn.BatchNorm2d(output_channels))
            # layers.append(nn.ReLU(inplace=True))
            layers.append(nn.GroupNorm(num_groups, output_channels, eps=1e-06, affine=True))
            layers.append(nn.SiLU())
        
        # layers.append(nn.AdaptiveAvgPool2d((output_resolution, output_resolution)))
        layers.append(nn.AdaptiveAvgPool2d((3*output_resolution, 2*output_resolution)))

        self.downsample_layers = nn.Sequential(*layers)
        
        # ## v2
        # # Define the layers for downsampling
        # layers = [
        #     # First downsampling
        #     nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=2, padding=1),
        #     nn.ReLU(),
        #     # # Second downsampling
        #     # nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=2, padding=1),
        #     # nn.ReLU(),
        #     # Adaptive pooling to reach the exact output size
        #     nn.AdaptiveAvgPool2d((output_resolution, output_resolution))
        # ]
        
        # # Combine all layers into a Sequential model
        # self.downsample_layers = nn.Sequential(*layers)
        
        
        ## ---- v3 -------
        #  # Initialize module list
        # modules = []
        # current_resolution = input_resolution
        # current_channels = input_channels

        # # Define the number of groups for GroupNorm, it must be a divisor of the number of channels
        # # We are choosing 32 because it's commonly used in practice and assuming the channels will be divisible by 32
        # num_groups = input_channels

        # # Adjust the number of channels to be divisible by the number of groups
        # if current_channels % num_groups != 0:
        #     adjusted_channels = (current_channels // num_groups + 1) * num_groups
        #     modules.append(nn.Conv2d(current_channels, adjusted_channels, kernel_size=1))
        #     modules.append(nn.GroupNorm(num_groups, adjusted_channels, eps=1e-06, affine=True))
        #     modules.append(nn.SiLU())
        #     current_channels = adjusted_channels

        # # Add downsampling layers
        # next_resolution = current_resolution
        # while next_resolution > output_resolution:
        #     current_resolution = next_resolution
            
        #     # Define a block with Conv2D, GroupNorm and SiLU (Swish) activation
        #     modules.append(nn.Conv2d(current_channels, current_channels * 2, kernel_size=3, stride=2, padding=1))
        #     modules.append(nn.GroupNorm(num_groups, current_channels * 2, eps=1e-06, affine=True))
        #     modules.append(nn.SiLU())

        #     # Update current resolution and channels
        #     # Calculate the next resolution, making sure it doesn't go below the target
        #     next_resolution = max(output_resolution, current_resolution // 2)
            
        #     current_channels *= 2

        # # If the number of channels doesn't match the target output_channels, add a 1x1 convolution
        # if current_channels != output_channels:
        #     modules.append(nn.Conv2d(current_channels, output_channels, kernel_size=1))
        #     modules.append(nn.GroupNorm(num_groups, output_channels, eps=1e-06, affine=True))
        #     modules.append(nn.SiLU())
    
        # print(f"current_resolution({current_resolution}) != output_resolution({output_resolution}) : {current_resolution != output_resolution}")
        # # If the resolution is still not matched (for non-power-of-2 scaling), use adaptive average pooling
        # if current_resolution != output_resolution:
        #     print("Init an adaptive pooling layer")
        #     modules.append(nn.AdaptiveAvgPool2d((output_resolution, output_resolution)))

        # self.downsample_layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.downsample_layers(x)
        # for ly in self.downsample_layers:
        #     print(f"---{ly._get_name()}---")
        #     print(f"input size:{x.shape}")
        #     x = ly(x)
        #     print(f"output size:{x.shape}")
        # st()
        # return x

class Interpolate(nn.Module):
    def __init__(self, size, mode='bilinear', align_corners=False):
        super(Interpolate, self).__init__()
        self.size = size
        self.mode = mode
        self.align_corners = align_corners if mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] else None

    def forward(self, x):
        x = F.interpolate(x, size=self.size, mode=self.mode, align_corners=self.align_corners)
        return x

# From: https://github.com/huggingface/diffusers/blob/v0.26.3/src/diffusers/models/autoencoders/autoencoder_kl.py#L35
class UNetDecoder(nn.Module):
    def __init__(self, vae, opt):
        super(UNetDecoder, self).__init__()
        self.vae = vae
        self.decoder = vae.decoder

        if opt.decode_splatter_to_128:
            
            # 1. no additional downsample layers
            if opt.decoder_upblocks_interpolate_mode is not None:

                if opt.decoder_upblocks_interpolate_mode == "last_layer":
                    self.decoder.up_blocks[-1].upsamplers = nn.ModuleList([]) 
                    self.decoder.up_blocks[-1].upsamplers.append(Interpolate(size=(128*3, 128*2), mode="nearest"))

                else:
                    find_interpolate_index={
                        "interpolate_upsample": 1,
                        "interpolate_downsample": 2,
                    }
                    interpolate_block_ind = find_interpolate_index[opt.decoder_upblocks_interpolate_mode]
                    for i, up_block in enumerate(self.decoder.up_blocks):
                        if i > interpolate_block_ind:
                            up_block.upsamplers = nn.ModuleList([]) 
                        elif i == interpolate_block_ind:
                            if opt.replace_interpolate_with_avgpool:
                                up_block.upsamplers = nn.ModuleList([nn.AdaptiveAvgPool2d((3*128, 2*128))]) 
                            else:
                                up_block.upsamplers = nn.ModuleList([Interpolate(size=(128*3, 128*2), mode="nearest")]) 
                            
            
                self.downsample_module = lambda x: x
                
            # 2-4. use additional downsample layers, not good
            elif opt.downsample_mode == "DownsampleModule":
                self.downsample_module = DownsampleModule(input_channels=14, output_channels=14, input_resolution=320, output_resolution=128,
                                                use_activation_at_downsample=opt.use_activation_at_downsample)
            elif opt.downsample_mode == "AvgPool":
                self.downsample_module = nn.AdaptiveAvgPool2d((3*128, 2*128))
            elif opt.downsample_mode == "ConvAvgPool":
                self.downsample_module = DownsampleModuleOnlyConvAvgPool(input_channels=14, output_channels=14, input_resolution=320, output_resolution=128,
                                                use_activation_at_downsample=opt.use_activation_at_downsample)
            else:
                assert NotImplementedError
        else:
            self.downsample_module = lambda x: x
        
        
      
        self.decoder = self.decoder.requires_grad_(False).eval()
        
        self.others = nn.Conv2d(128, 11, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # print(self.decoder)
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
        for i, up_block in enumerate(self.decoder.up_blocks):
            # print(f"{i}th upblock input: {sample.shape}")
            sample = up_block(sample, latent_embeds)
        
        # print(f"{i}th upblock output: {sample.shape}")
        # st()
        
        sample = self.decoder.conv_norm_out(sample)
        sample = self.decoder.conv_act(sample)
        rgb = self.decoder.conv_out(sample)
        others = self.others(sample)
        
        splatters_320 = torch.cat([others, rgb], dim=1)
        splatters_128 = self.downsample_module(splatters_320)
        # print(f"splatters_320:{splatters_320.shape}")
        # print(f"splatters_128:{splatters_128.shape}")
        # st()
        return splatters_128
        # return rgb
        
        
class Zero123PlusGaussianCodeUnet(nn.Module):
    def __init__(
        self,
        opt: Options,
    ):
        super().__init__()
        self.opt = opt

        # Load zero123plus model
        import sys
        sys.path.append('./zero123plus')
        
        # TODO:[BEGIN] check this part about requires grad -----------


        self.pipe = DiffusionPipeline.from_pretrained(
            opt.model_path,
            custom_pipeline=opt.custom_pipeline
        ).to('cuda')

        self.pipe.prepare() 
        self.vae = self.pipe.vae.requires_grad_(False).eval()
        self.vae.decoder.requires_grad_(False).eval() 
    
        print("Unet is trainable")
        self.unet = self.pipe.unet.requires_grad_(True).train() 
        
        if opt.scheduler_type == "cosine":
            # Assuming self.pipe.scheduler.config is your original FrozenDict configuration
            original_config = dict(self.pipe.scheduler.config)

            # Define the parameters that DDPMScheduler accepts
            ddpm_params = {
                'num_train_timesteps': original_config.get('num_train_timesteps', 1000),
                'beta_start': original_config.get('beta_start', 0.00085),
                'beta_end': original_config.get('beta_end', 0.012),
                'beta_schedule': 'linear'  # explicitly set to 'cosine'
            }

            # Create the new scheduler with the updated configuration
            # new_scheduler = DDPMScheduler(**ddpm_params)
            from diffusers import LMSDiscreteScheduler
            new_scheduler = LMSDiscreteScheduler(**ddpm_params)
            self.pipe.scheduler = new_scheduler
            print("We are using cosine scheduler")
            # st()
        else:
            self.pipe.scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config) # num_train_timesteps=1000

        self.decoder = UNetDecoder(self.vae, opt)
        self.decoder.requires_grad_(False).eval()
        
        # TODO:[END] check this part about requires grad -----------
    
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
        if self.opt.decoder_mode == "v1_fix_rgb":
            self.rgb_act = lambda x: x
        elif self.opt.decoder_mode == "v1_fix_rgb_remove_unscale":
            self.rgb_act = lambda x: unscale_image(x)
        else:
            self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5 # NOTE: may use sigmoid if train again

        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)
        
        ## specific for code optimization
        self.code_size = (self.pipe.unet.config.in_channels, 3 * opt.latent_resolution, 2 * opt.latent_resolution) # 4, 120, 80
        splatter_resolution = 8 * opt.latent_resolution
        splatter_channels = 14 # directly using xyz without offset. Otherwise 15
        self.splatter_size = (splatter_channels, splatter_resolution, splatter_resolution)
       
        if opt.init_from_mean:
            self.register_buffer('init_code', torch.zeros(self.code_size))
        else:
            self.init_code = None
        
        self.init_scale=1e-4
        self.mean_scale=1.0

        if opt.use_tanh_code_activation:
            print(f"[WARN]: USE tanh for code activation, which is not good for diffusion training")
            self.code_activation = lambda x: torch.tanh(x)
            self.code_activation_inverse = lambda x: torch.atanh(x)
        else:
            self.code_activation = lambda x: x
            self.code_activation_inverse = lambda x: x
      
    
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
        # print(f"t={t}, alpha_t, sigma_t:{alpha_t, sigma_t}")
        noise_pred = predict_noise0_diffuser(
            self.unet, noisy_latents, text_embeddings, t=t,
            guidance_scale=guidance_scale, cross_attention_kwargs=cross_attention_kwargs, 
            scheduler=scheduler, model=model
        )
        # return (noisy_latents - noise_pred * sigma_t) / alpha_t
        return (noisy_latents - noise_pred * sigma_t.reshape(-1, 1, 1, 1)) / alpha_t.reshape(-1, 1, 1, 1)
    
    def predict_eps(self, noisy_latents, text_embeddings, t, 
            guidance_scale=1.0, cross_attention_kwargs={}, 
            scheduler=None, lora_v=False, model='sd'):
        alpha_t, sigma_t = self.get_alpha(scheduler, t, noisy_latents.device)
        # print(f"t={t}, alpha_t, sigma_t:{alpha_t, sigma_t}")
        noise_pred = predict_noise0_diffuser(
            self.unet, noisy_latents, text_embeddings, t=t,
            guidance_scale=guidance_scale, cross_attention_kwargs=cross_attention_kwargs, 
            scheduler=scheduler, model=model
        )
        x = (noisy_latents - noise_pred * sigma_t.reshape(-1, 1, 1, 1)) / alpha_t.reshape(-1, 1, 1, 1)
        return {"noise_pred": noise_pred, "x0": x}
      

    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict
    
    def get_init_code_(self, num_scenes=None, device=None):
        code_ = torch.empty(
            self.code_size if num_scenes is None else (num_scenes, *self.code_size),
            device=device, requires_grad=True, dtype=torch.float32)
        if self.init_code is None:
            code_.data.uniform_(-self.init_scale, self.init_scale)
        else:
            code_.data[:] = self.code_activation.inverse(self.init_code * self.mean_scale)
        return code_
    
    def get_init_code_from_0123_encoder(self, images, num_scenes=None, device=None):
        
        if num_scenes is None: # images: [6, 3, 320, 320] 
            images = images[None]
        assert images.dim() == 5 # to contain the batch dim
        
        # make input 6 views into a 3x2 grid
        images = einops.rearrange(images, 'b (h2 w2) c h w -> b c (h2 h) (w2 w)', h2=3, w2=2) 
        # init code from pretrained zero123++ encoder
        code_ = self.encode_image(images) # [b, 4, 120, 80]
        if num_scenes is None:
            code_ = code_.squeeze(0)
        return code_
    
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
    
    def load_scenes(self, code_dir, data, eval_mode=False):
        code_list_ = []
        # optimizer_state_list = []
    
        device = data['cond'].device

        resume_dir = os.path.dirname(self.opt.resume)
        if "epoch" in os.path.basename(resume_dir):
            resume_dir = os.path.dirname(resume_dir)
        resume_code_dir = os.path.join(resume_dir, "code_dir")
        
        for i, scene_name in enumerate(data['scene_name']):
            # cache_file = os.path.join(code_dir, scene_name + '.pth')
            
            # cache_file_from_resume = ""
            # assert self.opt.resume is not None
            
            
            cache_file_from_resume = os.path.join(resume_code_dir, scene_name + '.pth')
                
            # if os.path.exists(cache_file):
            #     if self.opt.verbose_main:
            #         print(f"Load scene: {scene_name}")
            #     out = torch.load(cache_file, map_location='cpu')
            #     assert out['scene_name'] == scene_name

            #     code_ = out['param']['code_']
            #     code_list_.append(code_.requires_grad_(True))

            #     if not eval_mode:
            #         optimizer_state = out['optimizer']
            #         optimizer_state_list.append(optimizer_state)
            
            assert os.path.exists(cache_file_from_resume)
            if self.opt.verbose_main:
                print(f"Load scene from resume: {scene_name}")
            out = torch.load(cache_file_from_resume, map_location='cpu')
            assert out['scene_name'] == scene_name

            code_ = out['param']['code_']
            code_list_.append(code_.requires_grad_(True))

            # if not eval_mode:
            #     optimizer_state = out['optimizer']
            #     optimizer_state_list.append(optimizer_state)
                
            # else:
            #     # do init
            #     if self.opt.verbose_main: 
            #         print(f"Init scene: {scene_name}")
            #     if self.opt.code_init_from_0123_encoder: # data['cond']: [B, 320, 320, 3]
            #         code_ = self.get_init_code_from_0123_encoder(data['input'][i])
            #     else:
            #         code_ = self.get_init_code_() # torch.Size([4, 120, 80])
            #     code_list_.append(code_.requires_grad_(True))
    
        # --- code ---
        codes_ = torch.stack(code_list_, dim=0).to(device)
        codes = self.code_activation(codes_)
        
        # if eval_mode:
        return codes
        
        # # --- code optimizers ---
        # code_optimizers = self.build_optimizer(code_list_)
        # for ind, optimizer_state_single in enumerate(optimizer_state_list):
        #     optimizer_set_state(code_optimizers[ind], optimizer_state_single)
        
        # return code_list_, codes, code_optimizers

    def save_scenes(self, save_dir, code_list_, scene_names, code_optimizer_list):
        os.makedirs(save_dir, exist_ok=True)

        # codes_ = self.code_activation_inverse(codes) #NOTE: no need for inverse activation, because the passed in code_list_ is already before activation
        codes_ = torch.stack(code_list_, dim=0)

        for scene_id, scene_name_single in enumerate(scene_names):
            if self.opt.verbose_main:
                print(f"Save scene: {scene_name_single}")
            results = dict(
                scene_name=scene_name_single,
                param=dict(
                    code_=codes_.data[scene_id].cpu(), # with "_" is already before activation
                    ),
                optimizer=code_optimizer_list[scene_id].state_dict(),
                )

            torch.save(results, os.path.join(save_dir, scene_name_single) + '.pth')

    
    # def forward_splatters_with_activation_train_unet(self, cond, latents):
    #     st()
      
    #     B = cond.shape[0]
        
    #     with torch.no_grad():
    #         text_embeddings, cross_attention_kwargs = self.pipe.prepare_conditions(cond, guidance_scale=4.0)
    #         cross_attention_kwargs_stu = cross_attention_kwargs
        
    #     # TODO: [BEGIN] training and loss on latent code
     
    #     if latents is None:
    #         raise ValueError("Latents must be provided for the diffusion training process")

    #     gt_latents = latents
    #     guidance_scale = 1.0
        
    #     t = torch.randint(0, self.pipe.scheduler.timesteps.max(), (B,), device=latents.device)
    #     noise = torch.randn_like(latents, device=latents.device)
    #     noisy_latents = self.pipe.scheduler.add_noise(latents, noise, t)
      
    #     pred = "eps"
    #     if pred == "x0":
    #         x = self.predict_x0(
    #             noisy_latents, text_embeddings, t=t, guidance_scale=guidance_scale, 
    #             cross_attention_kwargs=cross_attention_kwargs, scheduler=self.pipe.scheduler, model='zero123plus')
        
    #         # do loss on x and gt_latents
    #         loss_latent = F.mse_loss(x, gt_latents)
    #     elif pred == "eps":
    #         res = self.predict_eps(
    #             noisy_latents, text_embeddings, t=t, guidance_scale=guidance_scale, 
    #             cross_attention_kwargs=cross_attention_kwargs, scheduler=self.pipe.scheduler, model='zero123plus')
    #         # do loss on x and gt_latents
    #         loss_latent = F.mse_loss(noise, res["noise_pred"])
    #         x = res["x0"]
        
        
    #     return loss_latent
        
        
        
    def forward_splatters_with_activation(self, images, cond, latents=None):
        B, V, C, H, W = images.shape
        # print(f"images.shape in forward+spaltter:{images.shape}") # SAME as the input_size
        with torch.no_grad():
            text_embeddings, cross_attention_kwargs = self.pipe.prepare_conditions(cond, guidance_scale=4.0)
            cross_attention_kwargs_stu = cross_attention_kwargs
        
        # TODO: [BEGIN] training and loss on latent code
     
        if latents is None:
            raise ValueError("Latents must be provided for the diffusion training process")


        gt_latents = latents
        guidance_scale = 1.0
        
        # cw
        # st()
        if self.opt.scheduler_type == "cosine":
            t = torch.randint(0, self.pipe.scheduler.timesteps.max().to(torch.int).item(), (B,), device=latents.device)
        else:
            t = torch.randint(0, self.pipe.scheduler.timesteps.max(), (B,), device=latents.device)
        noise = torch.randn_like(latents, device=latents.device)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, t)
          
        # print(noisy_latents.shape)
        # st()
    
        pred = "eps"
        if pred == "x0":
            x = self.predict_x0(
                noisy_latents, text_embeddings, t=t, guidance_scale=guidance_scale, 
                cross_attention_kwargs=cross_attention_kwargs, scheduler=self.pipe.scheduler, model='zero123plus')
        
            # do loss on x and gt_latents
            # add w(t) for loss: only for predict x0, not for v_pred or eps
            loss_latent = F.mse_loss(x, gt_latents)
        elif pred == "eps":
            res = self.predict_eps(
                noisy_latents, text_embeddings, t=t, guidance_scale=guidance_scale, 
                cross_attention_kwargs=cross_attention_kwargs, scheduler=self.pipe.scheduler, model='zero123plus')
            # do loss on x and gt_latents
            loss_latent = F.mse_loss(noise, res["noise_pred"])
            x = res["x0"]
     
        # TODO: [END] training and loss on latent code
        
        x = self.decode_latents(x) # (B, 14, H, W)
        
        # TODO: whether need the following gaussian prediction and 

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
        return splatters, loss_latent
    
    
    
    def gs_weighted_mse_loss(self, pred_splatters, gt_splatters):
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
        
        
    def forward(self, data, step_ratio=1, splatter_guidance=False):
        # Gaussian shape: (B*6, 14, H, W)
        
        results = {}
        loss = 0
       
        # 1. optimize the splatters from the code
        if self.opt.codes_from_encoder:
            # codes=None
             # make input 6 views into a 3x2 grid
            images = einops.rearrange(data['input'], 'b (h2 w2) c h w -> b c (h2 h) (w2 w)', h2=3, w2=2) 
            # init code from pretrained zero123++ encoder
            data['codes']=self.encode_image(images)
        else:
            if 'codes' not in data:
                st()
        
            
        pred_splatters, loss_latent = self.forward_splatters_with_activation(data['input'],  data['cond'], latents=data['codes']) # [B, N, 14] # (B, 6, 14, H, W) # [B, H, W, 3], condition image: [1, 320, 320, 3]
        
        results['loss_latent'] = loss_latent
        
        results['splatters_from_code'] = pred_splatters # [1, 6, 14, 256, 256]
        if self.opt.verbose_main:
            print(f"model splatters_pred: {pred_splatters.shape}")
        
        
        # NOTE: when optimizing the splatter and code together, we do not need the loss from gt splatter images
        if self.opt.lambda_splatter > 0 and splatter_guidance:
            
            # print(f"Splatter guidance epoch. Use splatters_to_optimize to supervise the code pred splatters")
            # gt_splatters =  data['splatters_to_optimize'] # [1, 6, 14, 128, 128]
            
            if self.opt.decode_splatter_to_128:
                gt_splatters =  data['splatters_output']
            else:
                gt_splatters_low_res =  data['splatters_output']
                gt_splatters =  torch.stack([F.interpolate(sp, self.splatter_size[-2:]) for sp in gt_splatters_low_res], dim=0)
            # print(f"gt splatter size:{gt_splatters.shape}")
           
            # NOTE: discard the below of downsampling pred, but use upsampling gt
            # if gt_splatters.shape[-2:] != pred_splatters.shape[-2:]:
            #     print("pred_splatters:", pred_splatters.shape)
            #     B, V, C, H, W, = pred_splatters.shape
            #     pred_splatters_gt_size = einops.rearrange(pred_splatters, "b v c h w -> (b v) c h w")
            #     pred_splatters_gt_size = F.interpolate(pred_splatters_gt_size, size=gt_splatters.shape[-2:], mode='bilinear', align_corners=False) # we move this to calculating splatter loss only, while we keep this high res splatter for rendering
            #     pred_splatters_gt_size = einops.rearrange(pred_splatters_gt_size, "(b v) c h w -> b v c h w", b=B, v=V)
            #     st()
                
            # else:
            #     pred_splatters_gt_size = pred_splatters
            
            gs_loss_mse_dict = self.gs_weighted_mse_loss(pred_splatters, gt_splatters)
            loss_mse = gs_loss_mse_dict['total']
        
            results['loss_splatter'] = loss_mse
            loss = loss + self.opt.lambda_splatter * loss_mse
            # also log the losses for each attributes
            results['gs_loss_mse_dict'] = gs_loss_mse_dict
         
        ## ------- splatter -> gaussian ------- 
        gaussians = fuse_splatters(pred_splatters) # this is the gaussian from code
        
        if self.training: # random bg for training
            bg_color = torch.rand(3, dtype=torch.float32, device=gaussians.device)
        else:
            
            if self.opt.data_mode == "srn_cars":
                bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device)
            else:
                bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device) * 0.5
        
        if self.opt.render_gt_splatter:
            print("Render GT splatter --> load splatters then fuse")
            gaussians = fuse_splatters(data['splatters_output'])
            # print("directly load fused gaussian ply")
            # gaussians = self.gs.load_ply('/home/xuyimeng/Repo/LGM/data/splatter_gt_full/00000-hydrant-eval_pred_gs_6100_0/fused.ply').to(pred_splatters.device)
            # gaussians = gaussians.unsqueeze(0)
            
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
            # use the other views for rendering and supervision
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

            ## ------- end render ----------
            
            # ----- FIXME: calculate psnr shoud be at the end -----
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
            results['psnr'] = psnr
        
          
        if self.opt.lambda_rendering > 0:
            loss_mse_rendering = F.mse_loss(pred_images, gt_images) + F.mse_loss(pred_alphas, gt_masks)
            results['loss_rendering'] = loss_mse_rendering
            loss +=  self.opt.lambda_rendering * loss_mse_rendering
            if self.opt.verbose_main:
                print(f"loss rendering (with alpha):{loss_mse_rendering}")
        elif self.opt.lambda_alpha > 0:
            loss_mse_alpha = F.mse_loss(pred_alphas, gt_masks)
            results['loss_alpha'] = loss_mse_alpha
            loss += self.opt.lambda_alpha * loss_mse_alpha
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
            loss += self.opt.lambda_lpips * loss_lpips
            if self.opt.verbose_main:
                print(f"loss lpips:{loss_lpips}")
        
        # ----- rendering [end] -----
        psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
        results['psnr'] = psnr
        results['loss'] = loss
        
        # #### 2. loss on the splatters to be optimized
        # if 'splatters_to_optimize' in data:
        #     loss_splatter_cache = 0
        #     splatters_to_optimize = data['splatters_to_optimize']
        #     assert (self.opt.lambda_lpips + self.opt.lambda_rendering) > 0
        #     gaussians_opt = fuse_splatters(splatters_to_optimize)

        #     gs_results_opt = self.gs.render(gaussians_opt, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
        #     pred_images_opt = gs_results_opt['image'] # [B, V, C, output_size, output_size]
        #     pred_alphas_opt = gs_results_opt['alpha'] # [B, V, 1, output_size, output_size]

        #     results['images_opt'] = pred_images_opt
        #     results['alphas_opt'] = pred_alphas_opt
            
        #     gt_images = data['images_output'] # [B, V, 3, output_size, output_size], ground-truth novel views
        #     gt_masks = data['masks_output'] # [B, V, 1, output_size, output_size], ground-truth masks
        #     gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)

        #     ## ------- end render ----------
            
        #     # ----- FIXME: calculate psnr shoud be at the end -----
        #     psnr_opt = -10 * torch.log10(torch.mean((pred_images_opt.detach() - gt_images) ** 2))
        #     results['psnr_opt'] = psnr_opt

        #     if self.opt.lambda_rendering > 0: # FIXME: currently using the same set of rendering loss for code and splatter cache
        #         loss_mse_rendering_opt = F.mse_loss(pred_images_opt, gt_images) + F.mse_loss(pred_alphas_opt, gt_masks)
        #         results['loss_rendering_opt'] = loss_mse_rendering_opt
        #         loss_splatter_cache = loss_splatter_cache + self.opt.lambda_rendering * loss_mse_rendering_opt
        #         if self.opt.verbose_main:
        #             print(f"loss rendering - splatter cache - (with alpha):{loss_mse_rendering_opt}")
        #     elif self.opt.lambda_alpha > 0:
        #         loss_mse_alpha_opt = F.mse_loss(pred_alphas_opt, gt_masks)
        #         results['loss_alpha_opt'] = loss_mse_alpha_opt
        #         loss_splatter_cache = loss_splatter_cache + self.opt.lambda_alpha * loss_mse_alpha_opt
        #         if self.opt.verbose_main:
        #             print(f"loss alpha - splatter cache - :{loss_mse_alpha_opt}")
                    
        #     if self.opt.lambda_lpips > 0:
        #         loss_lpips_opt = self.lpips_loss(
        #             F.interpolate(gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
        #             F.interpolate(pred_images_opt.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
        #         ).mean()
        #         results['loss_lpips_opt'] = loss_lpips_opt
        #         loss_splatter_cache = loss_splatter_cache + self.opt.lambda_lpips * loss_lpips_opt
        #         if self.opt.verbose_main:
        #             print(f"loss lpips:{loss_lpips_opt}")
                    
        #     results['loss_splatter_cache'] = loss_splatter_cache
        # else:
        #     assert False
        
        return results
    