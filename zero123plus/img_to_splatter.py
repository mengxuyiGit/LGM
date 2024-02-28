import torch
import requests
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, DDPMScheduler, DDIMScheduler
import numpy 
from tqdm import tqdm
import rembg
import einops

# Load the pipeline
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.1", custom_pipeline="/mnt/kostas-graid/sw/envs/chenwang/workspace/diffgan/training/modules/zero123plus.py",
    torch_dtype=torch.float16
)

# Feel free to tune the scheduler!
# `timestep_spacing` parameter is not supported in older versions of `diffusers`    
# so there may be performance degradations
# We recommend using `diffusers==0.20.2`
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)
# pipeline.scheduler = DDIMScheduler.from_config(
#     pipeline.scheduler.config
# )
pipeline.to('cuda:0')

# Download an example image.
# cond = Image.open(requests.get("https://d.skis.ltd/nrp/sample-data/lysol.png", stream=True).raw)
# cond = [Image.open('extinguisher.png'), Image.open(requests.get("https://d.skis.ltd/nrp/sample-data/lysol.png", stream=True).raw)]
# cond = Image.open('extinguisher.png')
# cond = Image.open('/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/owl.png')
cond = Image.open('/mnt/kostas-graid/sw/envs/chenwang/workspace/instant123-old/training-runs/zero123plus/zero123plus-gpus1-batch1-same-vsd-20240223-222021-cond0_t950/007000-cond.png')

def to_rgb_image(maybe_rgba: Image.Image):
    if maybe_rgba.mode == 'RGB':
        return maybe_rgba
    elif maybe_rgba.mode == 'RGBA':
        rgba = maybe_rgba
        img = numpy.random.randint(127, 128, size=[rgba.size[1], rgba.size[0], 3], dtype=numpy.uint8)
        img = Image.fromarray(img, 'RGB')
        img.paste(rgba, mask=rgba.getchannel('A'))
        return img
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)

# for img in cond:
img = to_rgb_image(cond)
cond = [img, img, img, img]


def unscale_latents(latents):
    latents = latents / 0.75 + 0.22
    return latents

def unscale_image(image):
    image = image / 0.5 * 0.8
    return image

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

import glob
import os
import numpy as np
from ipdb import set_trace as st

import sys
sys.path.append('/home/xuyimeng/Repo/LGM')
from core.options import AllConfigs
import tyro
import cv2
import torch.nn.functional as F
# paths = sorted(glob.glob('/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/**/000.png'))
# paths = ['/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/0a9b36d36e904aee8b51e978a7c0acfd/000.png']
# paths = ['/mnt/kostas-graid/sw/envs/chenwang/workspace/instant123/logs/test/images/dora.png']
# output_path = '/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999-zero123plus1'

# scenes = sorted(os.listdir('/mnt/kostas-graid/sw/envs/chenwang/data/gso/gso_eval_gsec'))
# paths = [f'/mnt/kostas-graid/sw/envs/chenwang/data/gso/gso_eval_gsec/{f}/000.png' for f in scenes]
# output_path = '/mnt/kostas-graid/sw/envs/chenwang/workspace/gsec_compare/zero123plus75'

data_path_rendering = "/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999"
scene_name = "b0bce5ad99d84befaf9159681c551051"
paths = [f"{os.path.join(data_path_rendering, scene_name)}/000.png"]

opt = tyro.cli(AllConfigs)

noise_level = opt.inference_noise_level
latents_from_encode = True

from datetime import datetime
time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
run_dir = f"runs/zerp123plus_inference/{time_str}-noise_level_{noise_level}"
if latents_from_encode:
    run_dir += "latents_from_encode"

if opt.resume is not None:
    from safetensors.torch import load_file
    from core.models_zero123plus_inference import Zero123PlusGaussianInference, gt_attr_keys, start_indices, end_indices, fuse_splatters
    model = Zero123PlusGaussianInference(opt).to('cuda:0')
    
    print(f"Resume from ckpt: {opt.resume}")
    if opt.resume.endswith('safetensors'):
        ckpt = load_file(opt.resume, device='cpu')
    else:
        ckpt = torch.load(opt.condresume, map_location='cpu')
        
     # tolerant load (only load matching shapes)
    # model.load_state_dict(ckpt, strict=False)
    state_dict = model.state_dict()
    for k, v in ckpt.items():
        if k in state_dict: 
            if state_dict[k].shape == v.shape:
                state_dict[k].copy_(v)
            else:
                print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
        else:
            print(f'[WARN] unexpected param {k}: {v.shape}')
    
    model = model.to(torch.float16)
    run_dir += "_my_decoder"
    
    ## PREPARE rendering
    import kiui
    from kiui.cam import orbit_camera
    cam_radius = 1.5
    fovy = 49.1
    # camera near plane
    znear = 0.5
    # camera far plane
    zfar = 2.5
    bg = 0.5
    
    tan_half_fov = np.tan(0.5 * np.deg2rad(fovy))
    proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
    proj_matrix[0, 0] = 1 / tan_half_fov
    proj_matrix[1, 1] = 1 / tan_half_fov
    proj_matrix[2, 2] = (zfar + znear) / (zfar - znear)
    proj_matrix[3, 2] = - (zfar * znear) / (zfar - znear)
    proj_matrix[2, 3] = 1


output_path = os.path.join(os.getcwd(), run_dir)
os.makedirs(output_path, exist_ok=True)


for path in tqdm(paths):
    name = path.split('/')[-2]
    os.makedirs(os.path.join(output_path, name), exist_ok=True)
    pipeline.prepare()
    # pipeline.requires_grad_(False).eval(0)
    guidance_scale = 4.0
    img = to_rgb_image(Image.open(path))
    img.save(os.path.join(output_path, f'{name}/cond.png'))
    cond = [img]
    st()
    prompt_embeds, cak = pipeline.prepare_conditions(cond, guidance_scale=4.0)
    pipeline.scheduler.set_timesteps(noise_level, device='cuda:0')
    timesteps = pipeline.scheduler.timesteps
    # latents = torch.randn([1, pipeline.unet.config.in_channels, 120, 80], device='cuda:0', dtype=torch.float16)
    latents = torch.randn([1, pipeline.unet.config.in_channels, 48, 32], device='cuda:0', dtype=torch.float16)
    latents_init = latents.clone().detach()
    
    
    
    # ######## ----- [BEGIN] ----- 
    # with torch.no_grad():
    #     for i, t in enumerate(timesteps):
    #         latent_model_input = torch.cat([latents] * 2)
    #         latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)

    #         # predict the noise residual
    #         noise_pred = pipeline.unet(
    #             latent_model_input,
    #             t,
    #             encoder_hidden_states=prompt_embeds,
    #             cross_attention_kwargs=cak,
    #             return_dict=False,
    #         )[0]

    #         # perform guidance
    #         if True:
    #             noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    #             noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    #         # noise_pred = predict_noise0_diffuser(pipeline.unet, latents, prompt_embeds, t, guidance_scale, cak, pipeline.scheduler)

    #         # compute the previous noisy sample x_t -> x_t-1
    #         latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        
     
    #     latents1 = unscale_latents(latents)
    #     image = pipeline.vae.decode(latents1 / pipeline.vae.config.scaling_factor, return_dict=False)[0]
    #     image = unscale_image(image)

    #     latents_init1 = unscale_latents(latents_init)
    #     image_init = pipeline.vae.decode(latents_init1 / pipeline.vae.config.scaling_factor, return_dict=False)[0]
    #     image_init = unscale_image(image_init)
        
    # # result = pipeline.image_processor.postprocess(image, output_type='pil')
    # # result_init = pipeline.image_processor.postprocess(image_init, output_type='pil')
    # # import pdb; pdb.set_trace();
    # mv_image = einops.rearrange((image[0].clip(-1,1)+1).cpu().numpy()*127.5, 'c (h2 h) (w2 w)-> (h2 w2) h w c', h2=3, w2=2).astype(np.uint8) 
    # for i, image in enumerate(mv_image):
    #     image = rembg.remove(image).astype(np.float32) / 255.0
    #     if image.shape[-1] == 4:
    #         image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
    #     Image.fromarray((image * 255).astype(np.uint8)).save(os.path.join(output_path, f'{name}/{i:03d}.png'))
    # # for i, img in enumerate(result):
    # #     img.save(os.path.join(output_path, f'{name}/6view.png'))
    # # np.save(os.path.join(output_path, f'{name}/6view-latents.npy'), latents_init.cpu().numpy())
    # # np.save(os.path.join(output_path, f'{name}/6view-z.npy'), latents.cpu().numpy())
    # ######## ----- [END] ----- 
    
    
    
    ## renderin splatter image 
    
    ## cam
    cam_poses = []
    images = []
    masks = []
    results = {}
    
    for vid in range(1, 7):
        # camera_path = path.replace('000.png', f'{vid:03d}.npy') # os.path.join(uid, f'{vid:03d}.npy')
        # cam = np.load(camera_path, allow_pickle=True).item()
        # c2w = orbit_camera(-cam['elevation'], cam['azimuth'], radius=cam['radius'])
        # c2w = torch.from_numpy(c2w)
        # c2w[:3, 3] *= cam_radius / 1.5
        # cam_poses.append(c2w)  

        
        image_path = path.replace('000.png', f'{vid:03d}.png') # os.path.join(uid, f'{vid:03d}.png')
        camera_path = path.replace('000.png', f'{vid:03d}.npy') # os.path.join(uid, f'{vid:03d}.npy')

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255 # [512, 512, 4] in [0, 1]
        image = torch.from_numpy(image)

        cam = np.load(camera_path, allow_pickle=True).item()
        c2w = orbit_camera(-cam['elevation'], cam['azimuth'], radius=cam['radius'])
        c2w = torch.from_numpy(c2w)
     
        c2w[:3, 3] *= cam_radius / 1.5 # 1.5 is the default scale
        
        image = image.permute(2, 0, 1) # [4, 512, 512]
        mask = image[3:4] # [1, 512, 512]
        image = image[:3] * mask + (1 - mask) * bg # [3, 512, 512], to white bg
        image = image[[2,1,0]].contiguous() # bgr to rgb

        images.append(image)
        masks.append(mask.squeeze(0))
        cam_poses.append(c2w)
    
    images = torch.stack(images, dim=0) # [V, C, H, W]
    masks = torch.stack(masks, dim=0) # [V, H, W]
    cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]
    
    # normalized camera feats as in paper (transform the first pose to a fixed position)
    transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
    cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]

    images_input = F.interpolate(images[:].clone(), size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
    cam_poses_input = cam_poses[:].clone()
    
     # opengl to colmap camera for gaussian renderer
    cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
    
    # cameras needed by gaussian rasterizer
    cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
    cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
    cam_pos = - cam_poses[:, :3, 3] # [V, 3]
    
    results['cam_view'] = cam_view
    results['cam_view_proj'] = cam_view_proj
    results['cam_pos'] = cam_pos
    results['input'] = images_input
    
    data = {k:v[None].to("cuda:0") for k, v in results.items()}
    
    ## finish loading data
    
    
    if latents_from_encode:
        images = data['input'].to(torch.float16)
        images = einops.rearrange(images, 'b (h2 w2) c h w -> b c (h2 h) (w2 w)', h2=3, w2=2) 
        latents = model.encode_image(images) # [1, 4, 48, 32]
    
    x = model.decode_latents(latents.to(torch.float16)) # perform unscale in this func alr
    x = x.permute(0, 2, 3, 1)
        
    pos = model.pos_act(x[..., :3]) # [B, N, 3]
    opacity = model.opacity_act(x[..., 3:4])
    scale = model.scale_act(x[..., 4:7])
    rotation = model.rot_act(x[..., 7:11])
    rgbs = model.rgb_act(x[..., 11:])

    splatters = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, N, 14]
    splatters = einops.rearrange(splatters, 'b (h2 h) (w2 w) c -> b (h2 w2) c h w', h2=3, w2=2) # (B, 6, 14, H, W)
   
    gaussians = fuse_splatters(splatters)
    bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device) * 0.5
    
    gs_results = model.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
    pred_images = gs_results['image'] # [B, V, C, output_size, output_size]
    # pred_alphas = gs_results['alpha'] # [B, V, 1, output_size, output_size]
    pred_images = pred_images.detach().cpu().numpy() # [B, V, 3, output_size, output_size]
    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
    kiui.write_image(f'{output_path}/splatter_img_render_{noise_level}.jpg', pred_images)
    
    
    # latents = model.encode_image(images) # [1, 4, 48, 32]
    ## pipeline.decode

    with torch.no_grad():
        latents1 = unscale_latents(latents)
        image = pipeline.vae.decode(latents1 / pipeline.vae.config.scaling_factor, return_dict=False)[0]
        image = unscale_image(image)

        # latents_init1 = unscale_latents(latents_init)
        # image_init = pipeline.vae.decode(latents_init1 / pipeline.vae.config.scaling_factor, return_dict=False)[0]
        # image_init = unscale_image(image_init)
        
    # result = pipeline.image_processor.postprocess(image, output_type='pil')
    # result_init = pipeline.image_processor.postprocess(image_init, output_type='pil')
    # import pdb; pdb.set_trace();
    mv_image = einops.rearrange((image[0].clip(-1,1)+1).cpu().numpy()*127.5, 'c (h2 h) (w2 w)-> (h2 w2) h w c', h2=3, w2=2).astype(np.uint8) 
    for i, image in enumerate(mv_image):
        image = rembg.remove(image).astype(np.float32) / 255.0
        if image.shape[-1] == 4:
            image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
        Image.fromarray((image * 255).astype(np.uint8)).save(os.path.join(output_path, f'{name}/_encoded_{i:03d}.png'))