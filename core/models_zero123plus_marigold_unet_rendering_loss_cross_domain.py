import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import kiui
from kiui.lpips import LPIPS
from diffusers import DiffusionPipeline, DDPMScheduler
import PIL
from PIL import Image
import einops

from core.options import Options
from core.gs import GaussianRenderer

from ipdb import set_trace as st
import matplotlib.pyplot as plt
import os


from core.dataset_v5_marigold import ordered_attr_list, attr_map, sp_min_max_dict
from pytorch3d.transforms import quaternion_to_axis_angle, axis_angle_to_quaternion
from diffusers.models.autoencoders.vae import DecoderOutput

def fuse_splatters(splatters):
    # fuse splatters
    B, V, C, H, W = splatters.shape
    
    x = splatters.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)
    
    # # SINGLE VIEW splatter 
    # x = splatters.permute(0, 1, 3, 4, 2)[:,0].reshape(B, -1, 14)
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
        # img = np.random.randint(127, 128, size=[rgba.size[1], rgba.size[0], 3], dtype=np.uint8)
        img = np.random.randint(255, 256, size=[rgba.size[1], rgba.size[0], 3], dtype=np.uint8)
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
    sp_image_o = sp_image_o.clip(0,1) 
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

        # from main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared import store_initial_weights, compare_weights
        
        if opt.use_video_decoderST:
            print("Load video pipe anyway")
            pipe_svd = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt",
            ).to('cuda')
            self.pipe.vae.decoder = pipe_svd.vae.decoder
            del pipe_svd
            # self.pipe_svd = pipe_svd
        else:
            print("not load pipe_svd")
        
        num_attributes = 5 if not opt.save_xyz_opacity_for_cascade else 2
        print("num_attributes is: ",num_attributes)
        self.pipe.prepare(random_init_unet=opt.random_init_unet, class_emb_cat=opt.class_emb_cat,
                          num_attributes=num_attributes) 
        self.vae = self.pipe.vae.requires_grad_(False).eval()
        self.unet = self.pipe.unet.requires_grad_(False).eval()

        
        if self.opt.decoder_with_domain_embedding:
            # # change the conv_in dim to 5
            # new_conv_in = nn.Conv2d(5,512,3, padding=(1,1)).requires_grad_(False)
            # new_conv_in.weight[:,:4].copy_(self.pipe.vae.decoder.conv_in.weight)
            # self.pipe.vae.decoder.conv_in = new_conv_in
            # # post_quant_conv
            # new_post_quant_conv = nn.Conv2d(5,5,1, padding=(1,1)).requires_grad_(False)
            # new_post_quant_conv.weight[:4,:4].copy_(self.pipe.vae.post_quant_conv.weight)
            # self.pipe.vae.post_quant_conv = new_post_quant_conv
            
            # init learnable embeddings for decoder
            # self.decoder_domain_embedding = nn.Parameter(torch.randn(5,16,16))
            if self.opt.decoder_domain_embedding_mode == "learnable":
                self.decoder_domain_embedding = nn.Parameter(torch.zeros(5,16,16))
            elif self.opt.decoder_domain_embedding_mode == "sincos":
                # add pos embedding on domain embedding
                one_hot = torch.eye(5)
                sin_cos = torch.cat([
                        torch.sin(one_hot),
                        torch.cos(one_hot)
                    ], dim=-1)
                sin_cos *= 0.1
                padded_e = F.pad(sin_cos,(3,3,0,0))  # 5, 16
                sqr_e = einops.rearrange(padded_e, "b (h w) -> b h w", h=4, w=4)  # 5,4,4
                repeat_e = sqr_e.repeat(1,4,4)
                self.register_buffer("decoder_domain_embedding", repeat_e)
                # self.decoder_domain_embedding = repeat_e
    
       
        self.pipe.scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config) # num_train_timesteps=1000, v_prediciton
        
        print(f"drop cond with prob {self.opt.drop_cond_prob}")
    
        # Gaussian Renderer
        self.gs = GaussianRenderer(opt)

        # # activations...
        # self.pos_act = lambda x: x.clamp(-1, 1)

        # if opt.scale_bias_learnable:
        #     self.scale_bias = nn.Parameter(torch.tensor([opt.scale_act_bias]), requires_grad=False)
        # else:
        #     self.scale_bias = opt.scale_act_bias
       
        # if self.opt.scale_act == "biased_exp":
        #     max_scale = self.opt.scale_clamp_max # in torch.log scale
        #     min_scale = self.opt.scale_clamp_min
        #     # self.scale_act = lambda x: torch.exp(x + self.scale_bias)
        #     self.scale_act = lambda x: torch.exp(torch.clamp(x + self.scale_bias, max=max_scale, min=min_scale))
        # elif self.opt.scale_act == "biased_softplus":
        #     max_scale = torch.exp(torch.tensor([self.opt.scale_clamp_max])).item() # in torch.log scale
        #     min_scale = torch.exp(torch.tensor([self.opt.scale_clamp_min])).item()
        #     # self.scale_act = lambda x: 0.1 * F.softplus(x + self.scale_bias)
        #     self.scale_act = lambda x: torch.clamp(0.1 * F.softplus(x + self.scale_bias), max=max_scale, min=min_scale)
        # elif self.opt.scale_act == "softplus":
        #     # self.scale_act = lambda x: 0.1 * F.softplus(x)
        #     max_scale = torch.exp(torch.tensor([self.opt.scale_clamp_max])).item() # in torch.log scale
        #     min_scale = torch.exp(torch.tensor([self.opt.scale_clamp_min])).item()
        #     self.scale_act = lambda x: torch.clamp(0.1 * F.softplus(x), max=max_scale, min=min_scale)
        # else: 
        #     raise ValueError ("Unsupported scale_act")
        
        # self.opacity_act = lambda x: torch.sigmoid(x)
        # self.rot_act = F.normalize
       
        # LPIPS loss
        # if self.opt.lambda_lpips > 0:
        self.lpips_loss = LPIPS(net='vgg')
        self.lpips_loss.requires_grad_(False)
        
        self.skip_decoding = (self.opt.lambda_rendering + self.opt.lambda_rendering + self.opt.lambda_splatter + self.opt.lambda_splatter_lpips) <= 0 and self.opt.train_unet
        if self.skip_decoding:
            print("Skip decoding the latents, save memory")
        

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
    
    
    def ST_decode(self, z: torch.Tensor,
        num_frames: int,
        return_dict: bool = True,
    ):
        """
        Decode a batch of images.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.

        """
        batch_size = z.shape[0] // num_frames
        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=z.dtype, device=z.device)
        decoded = self.pipe.vae.decoder(z, num_frames=num_frames, image_only_indicator=image_only_indicator)
        # decoded = self.pipe_svd.vae.decoder(z, num_frames=num_frames, image_only_indicator=image_only_indicator)

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)
    
    
    def forward(self, data, step_ratio=1, splatter_guidance=False, save_path=None, prefix=None, get_decoded_gt_latents=False):
        # Gaussian shape: (B*6, 14, H, W)

        results = {}
        loss = 0
      
        # encoder input: all the splatter attr pngs 
        images_all_attr_list = []
        if self.opt.train_unet_single_attr is not None:
            ordered_attr_list_local = self.opt.train_unet_single_attr
        # elif self.opt.finetune_decoder and self.opt.finetune_decoder_single_attr is not None:
        #     ordered_attr_list_local = self.opt.finetune_decoder_single_attr
        else:
            ordered_attr_list_local = ordered_attr_list
            
        for attr_to_encode in ordered_attr_list_local:
            sp_image = data[attr_to_encode]
            # print(f"[data]{attr_to_encode}: {sp_image.min(), sp_image.max()}")
            images_all_attr_list.append(sp_image)
        images_all_attr_batch = torch.stack(images_all_attr_list)
    
        A, B, _, _, _ = images_all_attr_batch.shape # [5, 1, 3, 384, 256]
        images_all_attr_batch = einops.rearrange(images_all_attr_batch, "A B C H W -> (B A) C H W")
        
        if save_path is not None:    
            images_to_save = images_all_attr_batch.detach().cpu().numpy() # [5, 3, output_size, output_size]
            images_to_save = (images_to_save + 1) * 0.5
            images_to_save = einops.rearrange(images_to_save, "a c (m h) (n w) -> (a h) (m n w) c", m=3, n=2)

        # do vae.encode
        sp_image_batch = scale_image(images_all_attr_batch)
        sp_image_batch = self.pipe.vae.encode(sp_image_batch).latent_dist.sample() * self.pipe.vae.config.scaling_factor
        latents_all_attr_encoded = scale_latents(sp_image_batch) # torch.Size([5, 4, 48, 32])
    
        if self.opt.custom_pipeline in ["./zero123plus/pipeline_v6_set.py", "./zero123plus/pipeline_v7_seq.py", 
                                        "./zero123plus/pipeline_v7_no_seq.py",
                                        "./zero123plus/pipeline_v8_cat.py"]:
            gt_latents = latents_all_attr_encoded
        elif self.opt.custom_pipeline in ["./zero123plus/pipeline_v2.py"] and self.opt.train_unet_single_attr is not None:
            gt_latents = latents_all_attr_encoded
            # print("[self.opt.train_unet_single_attr]: gt_latents = ", gt_latents.shape)
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
            
            if self.opt.decoder_with_domain_embedding:
                decoder_domain_embedding = self.decoder_domain_embedding.unsqueeze(1).repeat(1,6,1,1)
                decoder_domain_embedding = einops.rearrange(decoder_domain_embedding, "a (m n) h w -> a 1 (m h) (n w)", m=3, n=2)
                latents_all_attr_to_decode = latents_all_attr_to_decode + decoder_domain_embedding.repeat(B,1,1,1)
                

        elif self.opt.train_unet:
            cond = data['cond'].unsqueeze(1).repeat(1,A,1,1,1).view(-1, *data['cond'].shape[1:]) 
            
            # unet 
            with torch.no_grad():
                # classifier-free guidance
                if np.random.rand() < self.opt.drop_cond_prob:
                    cond = torch.zeros_like(cond)
               
                text_embeddings, cross_attention_kwargs = self.pipe.prepare_conditions(cond, guidance_scale=1.0)
                cross_attention_kwargs_stu = cross_attention_kwargs        
        
            # same t for all domain
            t = torch.randint(0, self.pipe.scheduler.timesteps.max(), (B,), device=latents_all_attr_encoded.device)
            t = t.unsqueeze(1).repeat(1,A)
            
            if self.opt.xyz_zero_t:
                t[:,0] = torch.min(10 * torch.ones_like(t[:,0]), t[:,0])
            elif self.opt.cascade_on_xyz_opacity:
                cond_t_max = torch.min(200 * torch.ones_like(t[:,0]), t[:,0])
                cond_t = []
                for max_val in cond_t_max:
                    if max_val > 0:
                        cond_t.append(torch.randint(max_val, (1,)))
                    else:
                        cond_t.append(torch.tensor([0]))
                cond_t = torch.cat(cond_t)
                # print(cond_t, cond_t.shape)
                t[:,0] = cond_t # xyz
                t[:,1] = cond_t # opacity
                
            # if self.opt.different_t_schedule is not None:
            #     schedule_offset = torch.tensor(self.opt.different_t_schedule, device=t.device).unsqueeze(0)
            #     new_t = t + schedule_offset
            #     t = torch.clamp(new_t, min=0, max=self.pipe.scheduler.timesteps.max())
            
            t = t.view(-1)
            # print("batch t=", t)
            
            if self.opt.fixed_noise_level is not None:
                t = torch.ones_like(t).to(t.device) * self.opt.fixed_noise_level
                print(f"fixed noise level = {self.opt.fixed_noise_level}")
            
            noise = torch.randn_like(gt_latents, device=gt_latents.device)
            noisy_latents = self.pipe.scheduler.add_noise(gt_latents, noise, t)
        
            domain_embeddings = torch.eye(5).to(noisy_latents.device)
            if self.opt.train_unet_single_attr is not None:
                domain_embeddings = domain_embeddings[:len(self.opt.train_unet_single_attr)]
            if self.opt.cd_spatial_concat:
                domain_embeddings = torch.sum(domain_embeddings, dim=0, keepdims=True) # feed all domains
            # add pos embedding on domain embedding
            domain_embeddings = torch.cat([
                    torch.sin(domain_embeddings),
                    torch.cos(domain_embeddings)
                ], dim=-1)
            domain_embeddings = domain_embeddings.unsqueeze(0).repeat(B,1,1).view(-1, *domain_embeddings.shape[1:]) # repeat for batch

            # v-prediction with unet: (B A) 4 48 32
            v_pred = self.unet(noisy_latents, t, encoder_hidden_states=text_embeddings, cross_attention_kwargs=cross_attention_kwargs, class_labels=domain_embeddings).sample

            # get v_target
            alphas_cumprod = self.pipe.scheduler.alphas_cumprod.to(
                device=noisy_latents.device, dtype=noisy_latents.dtype
            )
            alpha_t = (alphas_cumprod[t] ** 0.5).view(-1, 1, 1, 1)
            sigma_t = ((1 - alphas_cumprod[t]) ** 0.5).view(-1, 1, 1, 1)
            v_target = alpha_t * noise - sigma_t * gt_latents # (B A) 4 48 32

            # calculate the latent loss of each attribute separately
            if self.opt.log_each_attribute_loss or (self.opt.lambda_each_attribute_loss is not None):
                v_pred_AB = einops.rearrange(v_pred, "(B A) C H W -> A B C H W", B=B, A=A)
                v_target_AB = einops.rearrange(v_target, "(B A) C H W -> A B C H W", B=B, A=A)
                l2_all = (v_target_AB - v_pred_AB) ** 2
                l2_each_attr = torch.mean(l2_all, dim=np.arange(1,l2_all.dim()).tolist())
                for l2_, attr_ in zip(l2_each_attr, ordered_attr_list_local):
                    results[f"loss_latent_{attr_}"] = l2_
                if self.opt.lambda_each_attribute_loss is not None:
                    # print(f"weighted_loss_splatter: {self.opt.lambda_each_attribute_loss}")
                    weighted_loss_splatter =  l2_each_attr * torch.tensor(self.opt.lambda_each_attribute_loss, device=l2_each_attr.device)
                    results["loss_latent"] = torch.mean(weighted_loss_splatter)
            else:
                loss_latent = F.mse_loss(v_pred, v_target)     
                results['loss_latent'] = loss_latent * self.opt.lambda_latent
            
            if self.skip_decoding and save_path is None:
                return results
                
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
                _alpha_t = einops.rearrange(alpha_t.flatten(), "(B A)-> B A", B=B, A=A)[:,0]
                _rendering_w_t = _alpha_t ** 2 # shape [B]
                # print(f"_rendering_w_t of {t}: ", _rendering_w_t)
            else:
                _rendering_w_t = 1
        
        # allow inference
        elif self.opt.inference_finetuned_unet and not get_decoded_gt_latents:
            with torch.no_grad():
                guidance_scale = self.opt.guidance_scale

                inference_unseen = False
                if inference_unseen:
                    import requests
                    # cond = to_rgb_image(Image.open(requests.get("https://d.skis.ltd/nrp/sample-data/lysol.png", stream=True).raw))
                    # name = "lysol"
                    cond = to_rgb_image(Image.open("/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/data_test/anya_rgba.png"))
                else:
                    cond = data['cond']
                    # cond = data['cond'].unsqueeze(1).repeat(1,A,1,1,1).view(-1, *data['cond'].shape[1:]) 
                
                prompt_embeds, cak = self.pipe.prepare_conditions(cond, guidance_scale=guidance_scale)
                if guidance_scale > 1.0:
                    prompt_embeds = torch.cat([prompt_embeds[0:1]]*gt_latents.shape[0] + [prompt_embeds[1:]]*gt_latents.shape[0], dim=0) # torch.Size([10, 77, 1024])
                    cak['cond_lat'] = torch.cat([cak['cond_lat'][0:1]]*gt_latents.shape[0] + [cak['cond_lat'][1:]]*gt_latents.shape[0], dim=0)
                
                print(f"cak: {cak['cond_lat'].shape}") # always 64x64, not affected by cond size
                self.pipe.scheduler.set_timesteps(30, device='cuda:0')
                
                timesteps = self.pipe.scheduler.timesteps
                debug = False
                if debug:
                    debug_t = torch.tensor(50, dtype=torch.int64, device='cuda:0',)
                    noise = torch.randn_like(gt_latents, device='cuda:0', dtype=torch.float32)
                    t = torch.ones((5,), device=gt_latents.device, dtype=torch.int)
                    latents = self.pipe.scheduler.add_noise(gt_latents, noise, t*debug_t)
                    
                    timesteps = [debug_t]
                else:
                    latents  = torch.randn_like(gt_latents, device='cuda:0', dtype=torch.float32)
                
                if self.opt.xyz_zero_t:
                    assert B==1
                    xyz_t = 10 * torch.ones((1,), device=gt_latents.device, dtype=torch.int)
                    gt_latents_xyz = gt_latents[:1]
                    noise_xyz = torch.randn_like(gt_latents_xyz, device='cuda:0', dtype=torch.float32)
                    latents_xyz = self.pipe.scheduler.add_noise(gt_latents_xyz, noise_xyz, xyz_t)
                
                domain_embeddings = torch.eye(5).to(latents.device)
                if self.opt.train_unet_single_attr is not None:
                    # TODO: get index of that attribute
                    domain_embeddings = domain_embeddings[:len(self.opt.train_unet_single_attr)]
                    
                if self.opt.cd_spatial_concat:
                    st()
                    domain_embeddings = torch.sum(domain_embeddings, dim=0, keepdims=True) # feed all domains
                domain_embeddings = torch.cat([
                        torch.sin(domain_embeddings),
                        torch.cos(domain_embeddings)
                    ], dim=-1)
                
                # # check weights
                # # Save the initial weights
                # # mode = "load" if os.path.exists( 'initial_unet_weights.pth') else "save"
                # mode = "load"
                # if mode == "save":
                #     print("save unet weights")
                #     initial_unet_weights = self.pipe.unet.state_dict()
                #     folder = "svd" if self.opt.use_video_decoderST else "sd"
                #     os.makedirs(folder, exist_ok=True)
                #     torch.save(initial_unet_weights, os.path.join(folder, 'initial_unet_weights.pth'))
                #     with open(os.path.join(folder, 'initial_unet.txt'), "w") as f:
                #         print(self.pipe.unet, file=f)
                #     st()
                # elif mode == "load":
                #     # Load the saved weights
                #     # folder = "sd" if self.opt.use_video_decoderST else "svd"
                #     # assert not self.opt.only_train_attention
                #     folder = "sd_old"
                #     saved_unet_weights = torch.load(os.path.join(folder, 'initial_unet_weights.pth'))

                #     # Get the current weights
                #     current_unet_weights = self.pipe.unet.state_dict()

                #     # Function to compare weights
                #     def compare_weights(saved_weights, current_weights):
                #         for key in saved_weights.keys():
                #             if not torch.equal(saved_weights[key], current_weights[key]):
                #                 print(f"Difference found in layer: {key}")
                #             else:
                #                 pass
                #                 print("---")
                #                 # print(f"No difference in layer: {key}")

                #     # Compare the weights
                #     compare_weights(saved_unet_weights, current_unet_weights)
                # # ------- end -------
                
                # cfg
                if guidance_scale > 1.0:
                    domain_embeddings = torch.cat([domain_embeddings]*2, dim=0)
                
                if self.opt.xyz_opacity_for_cascade_dir is not None:
                    print("Loading pre-saved xyz_opacity    ")
                    gt_latents_xyz_opacity = torch.load(f"{self.opt.xyz_opacity_for_cascade_dir}/{prefix}_xyz_opacity.pt")
                    gt_latents[:2] = gt_latents_xyz_opacity
                else:
                    gt_latents_xyz_opacity = gt_latents[:2]
            
                # latents_init = latents.clone().detach()
                for _, t in enumerate(timesteps):
                    print(f"enumerate(timesteps) t={t}")
                    if self.opt.xyz_zero_t and t >= 10: 
                        print("use t=10 for latent-xyz")
                        latents[:1] = latents_xyz
                    elif self.opt.cascade_on_xyz_opacity:
                        # make sure the xyz and opacity has some noise but smaller than the others
                        cascade_t = max(min(100, t-100), 0) * torch.ones((1,), device=gt_latents.device, dtype=torch.int)
                        print(f"cascade_t: {cascade_t}")
                        noise_xyz_opacity = torch.randn_like(gt_latents_xyz_opacity, device='cuda:0', dtype=torch.float32)
                        latents_xyz_opacity = self.pipe.scheduler.add_noise(gt_latents_xyz_opacity, noise_xyz_opacity, cascade_t)
                        assert B==1
                        latents[:2] = latents_xyz_opacity
                    
                    if guidance_scale > 1.0:
                        latent_model_input = torch.cat([latents] * 2)
                    else:
                        latent_model_input = latents
                        t = t.repeat(A)
                    latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cak,
                        return_dict=False,
                        class_labels=domain_embeddings,
        
                    )[0]    

                    # perform guidance
                    if guidance_scale > 1.0:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    # compute the previous noisy sample x_t -> x_t-1
                    if debug:
                        alphas_cumprod = self.pipe.scheduler.alphas_cumprod.to(
                            device=latents.device, dtype=latents.dtype
                        )
                        alpha_t = (alphas_cumprod[t] ** 0.5).view(-1, 1, 1, 1)
                        sigma_t = ((1 - alphas_cumprod[t]) ** 0.5).view(-1, 1, 1, 1)
                        noise_pred = latents * sigma_t.view(-1, 1, 1, 1) + noise_pred * alpha_t.view(-1, 1, 1, 1)
                        latents = (latents - noise_pred * sigma_t) / alpha_t
                    else:
                        latents = self.pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                    
                print(latents.shape)
                if self.opt.cd_spatial_concat: # reshape back
                    latents = einops.rearrange(latents, " B C (A H) (m n W) -> (B A) C (m H) (n W)", B=data['cond'].shape[0], A=5, m=3, n=2)
                    # latents = einops.rearrange(latents, " B C (A H) (m n W) -> (B A) C (m H) (n W)", A=5, m=3, n=2)
                
                if self.opt.save_xyz_opacity_for_cascade:
                    torch.save(latents, f"{save_path}/{prefix}_xyz_opacity.pt")
                    return 
                # if cascade, replace xyz and opacity with gt
                if self.opt.cascade_on_xyz_opacity:
                    latents[:2] = gt_latents_xyz_opacity
                    print(f"use gt latent for cascade")
                
                # latents[1:2] = gt_latents[1:2]
                # print(f"use gt scale for inference unet")
                    
                latents_all_attr_to_decode = latents
                _rendering_w_t = 1

                

        elif self.opt.inference_finetuned_decoder or get_decoded_gt_latents: # NOTE: this condition must be check at last
            latents_all_attr_to_decode = gt_latents
            _rendering_w_t = 1
        
        else:
           raise NotImplementedError

        # vae.decode (batch process)
        latents_all_attr_to_decode = unscale_latents(latents_all_attr_to_decode)
        if self.opt.use_video_decoderST:
            print("video")
            image_all_attr_to_decode = self.ST_decode(latents_all_attr_to_decode / self.pipe.vae.config.scaling_factor, num_frames=5, return_dict=False)[0]
        else:
            image_all_attr_to_decode = self.pipe.vae.decode(latents_all_attr_to_decode / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
        image_all_attr_to_decode = unscale_image(image_all_attr_to_decode) # (B A) C H W 
        # THIS IS IMPORTANT!! Otherwise the very small negative value will overflow when * 255 and converted to uint8
        image_all_attr_to_decode = image_all_attr_to_decode.clip(-1,1)
        
        # L2 loss on the decoded splatter image, BOTH are within range [-1,1]
        # images_all_attr_batch.shape
        # loss_splatter = self.opt.lambda_splatter * F.mse_loss(image_all_attr_to_decode, images_all_attr_batch)
        loss_splatter = self.opt.lambda_splatter * F.l1_loss(images_all_attr_batch, image_all_attr_to_decode)
        results["loss_splatter"] = loss_splatter 
        if self.opt.lambda_splatter_lpips > 0:
            loss_splatter_lpips = self.lpips_loss(
                images_all_attr_batch, # gt, alr in [-1,1]
                image_all_attr_to_decode, # pred, alr in [-1,1]
            ).mean()
            results['loss_splatter_lpips'] = loss_splatter_lpips
        
      
        # Reshape image_all_attr_to_decode from (B A) C H W -> A B C H W and enumerate on A dim
        image_all_attr_to_decode = einops.rearrange(image_all_attr_to_decode, "(B A) C H W -> A B C H W", B=B, A=A)
        if self.opt.log_each_attribute_loss or (self.opt.lambda_each_attribute_loss is not None):
            images_all_attr_batch_AB = einops.rearrange(images_all_attr_batch, "(B A) C H W -> A B C H W", B=B, A=A)
            l2_all = (image_all_attr_to_decode - images_all_attr_batch_AB) ** 2
            l2_each_attr = torch.mean(l2_all, dim=np.arange(1,l2_all.dim()).tolist())
            for l2_, attr_ in zip(l2_each_attr, ordered_attr_list_local):
                results[f"loss_{attr_}"] = l2_
            
            if self.opt.lambda_each_attribute_loss is not None:
                # print(f"weighted_loss_splatter: {self.opt.lambda_each_attribute_loss}")
                weighted_loss_splatter =  l2_each_attr * torch.tensor(self.opt.lambda_each_attribute_loss, device=l2_each_attr.device)
                results["loss_splatter"] = torch.mean(weighted_loss_splatter)
        
        if self.opt.lambda_rendering <= 0 and self.opt.train_unet and save_path is None:
            return results

        # debug = False
        # if debug:
        #     image_all_attr_to_decode = einops.rearrange(images_all_attr_batch, "(B A) C H W -> A B C H W ", B=B, A=A)
        
        # decode latents into attrbutes again
        decoded_attr_list = []
        for i, _attr in enumerate(ordered_attr_list_local):
            batch_attr_image = image_all_attr_to_decode[i]
            # print(f"[vae.decode before]{_attr}: {batch_attr_image.min(), batch_attr_image.max()}")
            decoded_attr = denormalize_and_activate(_attr, batch_attr_image) # B C H W
            decoded_attr_list.append(decoded_attr)
            # print(f"[vae.decode after]{_attr}: {decoded_attr.min(), decoded_attr.max()}")

        if save_path is not None:
            images_to_save_encode = images_to_save
            decoded_attr_3channel_image_batch = einops.rearrange(image_all_attr_to_decode, "A B C H W -> (B A) C H W ", B=B, A=A)
            images_to_save = decoded_attr_3channel_image_batch.to(torch.float32).detach().cpu().numpy() # [5, 3, output_size, output_size]
            images_to_save = (images_to_save + 1) * 0.5
            images_to_save = einops.rearrange(images_to_save, "a c (m h) (n w) -> (a h) (m n w) c", m=3, n=2)
            # kiui.write_image(f'{save_path}/{prefix}images_all_attr_batch_decoded.jpg', images_to_save)
            # if A ==1:
            #     images_to_save = torch.cat([images_to_save_encode, images_to_save], dim=0)
            #     kiui.write_image(f'{save_path}/{prefix}single_attr_batch_decoded.jpg', images_to_save)
            # else:
            images_to_save = np.concatenate([images_to_save_encode, images_to_save], axis=1)
            kiui.write_image(f'{save_path}/{prefix}images_batch_attr_Lencode_Rdecoded.jpg', images_to_save)
            if self.opt.save_cond:
                # also save the cond image: cond 0-255, uint8
                if isinstance(cond, PIL.Image.Image):
                    cond.save(f'{save_path}/{prefix}cond.jpg')
                else:
                    cond_save = einops.rearrange(cond, "b h w c -> (b h) w c")
                    Image.fromarray(cond_save.cpu().numpy()).save(f'{save_path}/{prefix}cond.jpg')
            

        if self.opt.train_unet_single_attr is not None:
            return results # not enough attr for gs rendering
            
        splatter_mv = torch.cat(decoded_attr_list, dim=1) # [B, 14, 384, 256]
        
        # ## reshape 
        splatters_to_render = einops.rearrange(splatter_mv, 'b c (h2 h) (w2 w) -> b (h2 w2) c h w', h2=3, w2=2) # [1, 6, 14, 128, 128]
        gaussians = fuse_splatters(splatters_to_render) # B, N, 14

        if get_decoded_gt_latents:
            results['gaussians_LGM_decoded'] = gaussians
            return results
        if self.opt.fancy_video or self.opt.render_video:
            results['gaussians_pred'] = gaussians
        
        if self.training: # random bg for training
            bg_color = torch.rand(3, dtype=torch.float32, device=gaussians.device)
        else:
            bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device) * self.opt.bg # white bg

        #  render & calculate rendering loss
        if self.training:
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
        

        if (save_path is not None) or self.opt.inference_finetuned_decoder or self.opt.inference_finetuned_unet:
            with torch.no_grad():
                # render LGM GT output 
                image_all_attr_to_decode = einops.rearrange(images_all_attr_batch, "(B A) C H W -> A B C H W ", B=B, A=A)
                decoded_attr_list = [] # decode latents into attrbutes again
                for i, _attr in enumerate(ordered_attr_list_local):
                    batch_attr_image = image_all_attr_to_decode[i]
                    # print(f"[vae.decode before]{_attr}: {batch_attr_image.min(), batch_attr_image.max()}")
                    decoded_attr = denormalize_and_activate(_attr, batch_attr_image) # B C H W
                    decoded_attr_list.append(decoded_attr)
                    # print(f"[vae.decode after]{_attr}: {decoded_attr.min(), decoded_attr.max()}")
                splatter_mv = torch.cat(decoded_attr_list, dim=1) # [B, 14, 384, 256]
                splatters_to_render = einops.rearrange(splatter_mv, 'b c (h2 h) (w2 w) -> b (h2 w2) c h w', h2=3, w2=2) # [1, 6, 14, 128, 128]
                gaussians = fuse_splatters(splatters_to_render) # B, N, 14
                
                gs_results_LGM = self.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)

                if self.opt.fancy_video or self.opt.render_video:
                    results['gaussians_LGM'] = gaussians

                results['images_pred_LGM'] = gs_results_LGM['image'] 
                results['alphas_pred_LGM'] = gs_results_LGM['alpha']
                
                psnr_LGM = -10 * torch.log10(torch.mean((gs_results_LGM['image'].detach() - gt_images) ** 2))
                results['psnr_LGM'] = psnr_LGM.detach()
                
                # calculate lpips
                loss_lpips_LGM = self.lpips_loss(
                    F.interpolate(gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
                    F.interpolate(gs_results_LGM['image'].view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
                ).mean()
                results['loss_lpips_LGM'] = loss_lpips_LGM
            
        ## ------- end render ----------

        if self.opt.lambda_rendering > 0:
        # if True:
            # loss_mse_rendering = F.mse_loss(pred_images, gt_images) + F.mse_loss(pred_alphas, gt_masks)
            # calculate using batch 
            loss_mse_rendering = (pred_images - gt_images) ** 2 + (pred_alphas - gt_masks) ** 2
            loss_mse_rendering = _rendering_w_t * torch.mean(loss_mse_rendering, dim=np.arange(1,loss_mse_rendering.dim()).tolist())
            loss_mse_rendering = loss_mse_rendering.mean()
            results['loss_rendering'] = loss_mse_rendering
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
        # if True:
            try:
                loss_lpips = self.lpips_loss(
                    F.interpolate(gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
                    F.interpolate(pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
                )
                loss_lpips = _rendering_w_t * torch.mean(einops.rearrange(loss_lpips.flatten(), "(B N) -> B N", B=B), dim=1)
                loss_lpips = loss_lpips.mean()
            except:
                loss_lpips = loss # torch.ones_like(loss)
                
            results['loss_lpips'] = loss_lpips
            loss += self.opt.lambda_lpips * loss_lpips
            if self.opt.verbose_main:
                print(f"loss lpips:{loss_lpips}")
            
                
        # Calculate metrics
        # TODO: add other metrics such as SSIM
        psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
        results['psnr'] = psnr.detach()
        
        if isinstance(loss, int):
            loss = torch.as_tensor(loss, device=psnr.device, dtype=psnr.dtype)
        results['loss'] = loss
        # print("loss: ", loss)
        
        return results
    