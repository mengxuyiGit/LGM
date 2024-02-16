
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
from core.models import LGM
from mvdream.pipeline_mvdream import MVDreamPipeline

from ipdb import set_trace as st
from PIL import Image
from core.provider_objaverse import ObjaverseDataset as Dataset
# from core.provider_objaverse_inference_xuyi import ObjaverseDataset as Dataset

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

opt = tyro.cli(AllConfigs)

# model
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

rays_embeddings = model.prepare_default_rays(device)

tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
proj_matrix[0, 0] = 1 / tan_half_fov
proj_matrix[1, 1] = 1 / tan_half_fov
proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
proj_matrix[2, 3] = 1

# load image dream
pipe = MVDreamPipeline.from_pretrained(
    "ashawkey/imagedream-ipmv-diffusers", # remote weights
    torch_dtype=torch.float16,
    trust_remote_code=True,
    # local_files_only=True,
)
pipe = pipe.to(device)

# load rembg
bg_remover = rembg.new_session()

# process function
def process(opt: Options, path):
    name = os.path.splitext(os.path.basename(path))[0]
    print(f'[INFO] Processing {path} --> {name}')
    os.makedirs(opt.workspace, exist_ok=True)

    input_image = kiui.read_image(path, mode='uint8')

    # bg removal
    carved_image = rembg.remove(input_image, session=bg_remover) # [H, W, 4]
    mask = carved_image[..., -1] > 0

    # recenter
    image = recenter(carved_image, mask, border_ratio=0.2)
    
    # generate mv
    image = image.astype(np.float32) / 255.0

    # rgba to rgb white bg
    if image.shape[-1] == 4:
        image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
    
    mode = 'use_dataloader'
    # mode = 'original'
    if mode in ['original', 'use_rendered']:
        if mode == 'original':
            mv_image = pipe('', image, guidance_scale=5.0, num_inference_steps=30, elevation=0)
            mv_image = np.stack([mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0) # [4, 256, 256, 3], float32

        elif mode == 'use_rendered':
            render_path = '/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/data-1000/0c77dfdf9430465f9767a58d56e8fca1'
            imgs = [np.array(Image.open(f'{render_path}/{i:03d}.png')) / 255.0 for i in range(1,5)]
            # imgs = [np.array(Image.open(f'/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/data-1000/0a97a6e5c2894bfba2d347d333756b0e/{i:03d}.png')) / 255.0 for i in range(1,5)]
            imgs = [img[..., :3] * img[..., 3:4] + (1 - img[..., 3:4]) for img in imgs]
            mv_image = np.stack(imgs, axis=0)
            name = render_path.split('/')[-1]
            print(name)
            # st()
    
        save_mv_image = True
        if save_mv_image and mode != 'use_dataloader':
            images_array_scaled = (mv_image * 255).astype('uint8')

            # Loop through each image in the scaled array
            for i in range(images_array_scaled.shape[0]):
                # Extract the ith image from the scaled array
                current_image = images_array_scaled[i]

                # Convert the NumPy array to an image
                image = Image.fromarray(current_image)

                # Save the image with a unique filename (e.g., image_0.png, image_1.png, ...)
                _im_name = os.path.join(opt.workspace, f'{name}_mvimage_{i}.png')
                image.save(_im_name)  

        # generate gaussians
        input_image = torch.from_numpy(mv_image).permute(0, 3, 1, 2).float().to(device) # [4, 3, 256, 256]
        input_image = F.interpolate(input_image, size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False)
        input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0) # [1, 4, 9, H, W]

    elif mode == 'use_dataloader':
        # render_path = '/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/data-1000/0c77dfdf9430465f9767a58d56e8fca1' # huahua
        render_path = '/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/0a9b36d36e904aee8b51e978a7c0acfd'
        
        test_dataset = Dataset(opt, name=render_path, training=False)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )

        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                print(i)
                for item in data:
                    try:
                        data[item] = data[item].to(device)
                    except:
                        pass
            
        input_image = data['input']
        # name = 'jiatao'
        name = 'jiatao_'+render_path.split('/')[-1]
        
    else:
        assert ValueError

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            # generate gaussians
            gaussians = model.forward_gaussians(input_image)
        
        # save gaussians
        model.gs.save_ply(gaussians, os.path.join(opt.workspace, name + '.ply'))
       

        # render 360 video 
        images = []
        elevation = 0

        # if opt.save_inference_img:
        if True:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                out = model(data)
            gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
            gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
            kiui.write_image(f'{opt.workspace}/inference_gt_images.jpg', gt_images)

            pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
            pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
            kiui.write_image(f'{opt.workspace}/inference_pred_images.jpg', pred_images)
            
            

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
if os.path.isdir(opt.test_path):
    file_paths = glob.glob(os.path.join(opt.test_path, "*"))
else:
    file_paths = [opt.test_path]
print(file_paths)

for path in file_paths:
    if os.path.isdir(path):
        continue
    process(opt, path)
    # st()
