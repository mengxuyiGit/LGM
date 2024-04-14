
import os
import tyro
import glob
import imageio
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file
import rembg

import kiui
from kiui.op import recenter
from kiui.cam import orbit_camera

from core.options import AllConfigs, Options
from core.models_zero123plus import Zero123PlusGaussian
from core.models_fix_pretrained import LGM

# from mvdream.pipeline_mvdream import MVDreamPipeline

from torchvision import transforms
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# Normalization typically required for pre-trained models
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Combine ToTensor and Normalize transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    # normalize
])

opt = tyro.cli(AllConfigs)

# model
if opt.model_type == 'Zero123PlusGaussian':
    model = Zero123PlusGaussian(opt)
elif opt.model_type == 'LGM':
    model = LGM(opt)

# resume pretrained checkpoint
if opt.resume is not None:
    if opt.resume.endswith('safetensors'):
        ckpt = load_file(opt.resume, device='cpu')
    else:
        ckpt = torch.load(opt.resume, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    print(f'[INFO] Loaded checkpoint from {opt.resume}')
else:
    print(f'[WARN] model randomly initialized, are you sure?')

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.half().to(device)
model.eval()

# rays_embeddings = model.prepare_default_rays(device)

tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
proj_matrix[0, 0] = 1 / tan_half_fov
proj_matrix[1, 1] = 1 / tan_half_fov
proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
proj_matrix[2, 3] = 1

# # load image dream
def generate_mv_image(image, pipe):
    mask = carved_image[..., -1] > 0
    # recenter
    image = recenter(carved_image, mask, border_ratio=0.2)
    # generate mv
    image = image.astype(np.float32) / 255.0
    # rgba to rgb white bg
    if image.shape[-1] == 4:
        image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
    mv_image = pipe('', image, guidance_scale=5.0, num_inference_steps=30, elevation=0)
    mv_image = np.stack([mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0) # [4, 256, 256, 3], float32
    return mv_image
# pipe = MVDreamPipeline.from_pretrained(
#     "ashawkey/imagedream-ipmv-diffusers", # remote weights
#     torch_dtype=torch.float16,
#     trust_remote_code=True,
#     # local_files_only=True,
# )
# pipe = pipe.to(device)

# # load rembg
# bg_remover = rembg.new_session()
from ipdb import set_trace as st
import einops
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
# pipe = DiffusionPipeline.from_pretrained(
#         "sudo-ai/zero123plus-v1.1", custom_pipeline="/mnt/kostas-graid/sw/envs/chenwang/workspace/diffgan/training/modules/zero123plus.py",
#         torch_dtype=torch.float32
#     )
pipe = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.1", custom_pipeline="sudo-ai/zero123plus-pipeline",
    torch_dtype=torch.float16
)
pipe.to('cuda:0')
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipe.scheduler.config, timestep_spacing='trailing'
)
output_path = f"{opt.workspace}/zero123plus/outputs_v3_inference_my_decoder"
pipe.prepare()
guidance_scale = 4.0


def prepare_default_rays( device, elevation, azimuth):
        
    from kiui.cam import orbit_camera
    from core.utils import get_rays

    # cam_poses = np.stack([
    #     orbit_camera(-30, 30, radius=self.opt.cam_radius),
    #     orbit_camera(20, 90, radius=self.opt.cam_radius),
    #     orbit_camera(-30, 150, radius=self.opt.cam_radius),
    #     orbit_camera(20, 210, radius=self.opt.cam_radius),
    #     orbit_camera(-30, 270, radius=self.opt.cam_radius),
    #     orbit_camera(20, 330, radius=self.opt.cam_radius),
    # ], axis=0) # [4, 4, 4]
    cams = [orbit_camera(ele, azi, radius=opt.cam_radius) for (ele, azi) in zip(elevation, azimuth)]
    cam_poses = np.stack(cams, axis=0)
    cam_poses = torch.from_numpy(cam_poses)

    rays_embeddings = []
    for i in range(cam_poses.shape[0]):
        rays_o, rays_d = get_rays(cam_poses[i], opt.input_size, opt.input_size, opt.fovy) # [h, w, 3]
        rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
        rays_embeddings.append(rays_plucker)

        ## visualize rays for plotting figure
        # kiui.vis.plot_image(rays_d * 0.5 + 0.5, save=True)

    rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous().to(device) # [V, 6, h, w]
    
    return rays_embeddings

# process function
def process(opt: Options, path):
    name = os.path.splitext(os.path.basename(path))[0]
    print(f'[INFO] Processing {path} --> {name}')
    os.makedirs(opt.workspace, exist_ok=True)

    # input_image = kiui.read_image(path, mode='uint8')
    # carved_image = rembg.remove(input_image, session=bg_remover) # [H, W, 4]

    from PIL import Image
    poses = [np.load(f'{path}/{i:03d}.npy', allow_pickle=True).item() for i in range(1, 56)]
    elevations, azimuths = [-pose['elevation'] for pose in poses], [pose['azimuth'] for pose in poses]
    
    # get imgs from pipeline
    # cond = Image.open(f'{path}/{0:03d}.png')
    # cond = Image.open(f'srn_car_white_truck.png')
    cond = Image.open(f'srn_car_yellow_front.png')
    cond.save(f"{opt.workspace}/cond.png")
    # local_image_path='/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/0a9b36d36e904aee8b51e978a7c0acfd/000.png'
    # cond = Image.open(local_image_path)
    
    from_pipe = True
    if from_pipe:
        ## pipe
        result = pipe(cond, num_inference_steps=75).images[0]
        result.save(f"{opt.workspace}/zero123plus.png")
    
        image = transform(result) # 3, 960, 640
        mv_image = einops.rearrange((image.clip(0,1)).cpu().numpy()*255, 'c (h2 h) (w2 w)-> (h2 w2) h w c', h2=3, w2=2).astype(np.uint8)
        
        # rembg
        mv_image_no_bg = rembg.remove(einops.rearrange(mv_image, 'b h w c -> h (b w) c')) 

        mask = mv_image_no_bg[..., -1] > 0

        # recenter
        # image = recenter(mv_image_no_bg, mask, border_ratio=0.2)
        
        # generate mv
        image = mv_image_no_bg.astype(np.float32) / 255.0

        # rgba to rgb white bg
        if image.shape[-1] == 4:
            image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
        
        mv_image = einops.rearrange(image, 'h (b w) c -> b h w c', b=6)
        kiui.write_image(f"{opt.workspace}/mv_image.png", einops.rearrange(mv_image, 'b h w c -> h (b w) c'))

        # elevations = [30, -20, 30, -20, 30, -20]
        # azimuths = [30, 90, 150, 210, 270, 330]

        elevations = [-30, 20, -30, 20, -30, 20]
        azimuths = [316, 16, 76, 136, 196, 256]

    else:
        ## directly read
        imgs = [np.array(Image.open(f'{path}/{i:03d}.png')) / 255.0 for i in range(1, 7)]
        imgs = [img[..., :3] * img[..., 3:4] + (1 - img[..., 3:4]) for img in imgs]  
        mv_image = np.stack(imgs, axis=0)
        kiui.write_image(f"{opt.workspace}/mv_image.png", einops.rearrange(mv_image, 'b h w c -> h (b w) c'))

    # cond = np.array(Image.open(f'{path}/000.png').resize((opt.input_size, opt.input_size)))
    # mask = cond[..., 3:4] / 255
    # cond = cond[..., :3] * mask + (1 - mask) * int(opt.bg * 255)

    # generate gaussians
    input_image = torch.from_numpy(mv_image).permute(0, 3, 1, 2).float().to(device) # [4, 3, 256, 256]
    input_image = F.interpolate(input_image, size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False)
    input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    # if opt.model_type == 'LGM':
    #     rays_embeddings = model.prepare_default_rays(device, elevations[:6], azimuths[:6])
    #     input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    #     input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0) # [1, 4, 9, H, W]
    
    rays_embeddings = prepare_default_rays(device, elevations[:6], azimuths[:6])
    # st()
    input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0) # [1, 4, 9, H, W]
    

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # generate gaussians
            
            gaussians = model.forward_gaussians(input_image) if opt.model_type == 'LGM' else model.forward_gaussians(input_image.unsqueeze(0), cond.astype(np.uint8))
        
        # save gaussians
        model.gs.save_ply(gaussians, os.path.join(opt.workspace, name + '.ply'))

        # render at gt poses
        for (i, (ele, azi)) in enumerate(zip(elevations[6:], azimuths[6:])):
            cam_poses = torch.from_numpy(orbit_camera(ele, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)
            cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
            
            # cameras needed by gaussian rasterizer
            cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
            cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
            cam_pos = - cam_poses[:, :3, 3] # [V, 3]

            image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
            out = (image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8)
            kiui.write_image(f'{opt.workspace}/{i+6:03d}.png', out[0])

        # render 360 video 
        images = []
        elevation = 0

        if opt.fancy_video:

            azimuth = np.arange(0, 720, 4, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):
                
                cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

                cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                
                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                scale = min(azi / 360, 1)

                image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=scale)['image']
                images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))
        else:
            azimuth = np.arange(0, 360, 2, dtype=np.int32)
            for azi in tqdm.tqdm(azimuth):
                
                cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

                cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                
                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
                images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))

        images = np.concatenate(images, axis=0)
        imageio.mimwrite(os.path.join(opt.workspace, name + '.mp4'), images, fps=30)


assert opt.test_path is not None
# if os.path.isdir(opt.test_path):
#     file_paths = glob.glob(os.path.join(opt.test_path, "*"))
# else:
#     file_paths = [opt.test_path]
file_paths = [opt.test_path]
for path in file_paths:
    process(opt, path)
