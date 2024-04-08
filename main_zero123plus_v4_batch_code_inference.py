import tyro
import time
import random

import torch
from core.options import AllConfigs
from core.models_zero123plus import Zero123PlusGaussian, gt_attr_keys, start_indices, end_indices, fuse_splatters
from core.models_zero123plus_code import Zero123PlusGaussianCode
from core.models_zero123plus_code_unet_lora import Zero123PlusGaussianCodeUnetLora

from core.models_fix_pretrained import LGM

from accelerate import Accelerator, DistributedDataParallelKwargs
from safetensors.torch import load_file
from core.dataset_v4_batch import ObjaverseDataset as Dataset

import kiui
from datetime import datetime
import torch.utils.tensorboard as tensorboard
import shutil, os

from ipdb import set_trace as st
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
import torch.nn.functional as F

import warnings
from accelerate.utils import broadcast
import re

import numpy
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
# from zero123plus.img_to_mv_v3_my_decoder import to_rgb_image, unscale_image, unscale_latents

import einops
import rembg
import requests

from pytorch3d.transforms import quaternion_to_axis_angle, axis_angle_to_quaternion

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

def normalize_to_target(source_tensor, target_tensor):
    # Calculate mean and standard deviation of source tensor
    source_mean = torch.mean(source_tensor)
    source_std = torch.std(source_tensor)

    # Calculate mean and standard deviation of target tensor
    target_mean = torch.mean(target_tensor)
    target_std = torch.std(target_tensor)

    # Normalize source tensor to target distribution
    normalized_tensor = (source_tensor - source_mean) / source_std * target_std + target_mean

    return normalized_tensor

def fuse_splatters(splatters):
    # fuse splatters
    B, V, C, H, W = splatters.shape

    x = splatters.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)
    return x

def main():    
    import sys

    # # Your additional path
    # # your_path = "/home/xuyimeng/Repo/LGM"
    # your_path = " /home/chenwang/xuyi_runs"

    # # Add your path to sys.path
    # sys.path.append(your_path)
   

    opt = tyro.cli(AllConfigs)
    
    if opt.set_random_seed:
        # Set a manual seed for reproducibility
        seed = 42
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )

    # model
    if opt.model_type == 'Zero123PlusGaussian':
        model = Zero123PlusGaussian(opt)
        from core.dataset_v4_batch import ObjaverseDataset as Dataset
    elif opt.model_type == 'Zero123PlusGaussianCode':
        model = Zero123PlusGaussianCode(opt)
        from core.dataset_v4_code import ObjaverseDataset as Dataset
    
    elif opt.model_type == 'Zero123PlusGaussianCodeUnetLora':
        model = Zero123PlusGaussianCodeUnetLora(opt)
        from core.dataset_v4_code import ObjaverseDataset as Dataset
    
    # Check the number of GPUs
    num_gpus = accelerator.num_processes
    if accelerator.is_main_process:
        print(f"Num gpus: {num_gpus}")
    if num_gpus <= 1:
        time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        # Pick output directory.
        prev_run_dirs = []
        outdir = opt.workspace
        if os.path.isdir(outdir):
            prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        time_str = f'{cur_run_id:05d}'
        accelerator.wait_for_everyone()
    
    # c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    # assert not os.path.exists(c.run_dir)

    
    # # gathered_info = accelerator.all_gather(info_to_share)
    # accelerator.wait_for_everyone()
    # time_str = ''.join(chr(int(item)) for item in time_tensor.tolist())
    # print(time_str)
    
    # # Use torch.distributed to broadcast the workspace to all processes
    # time_str = torch.tensor(time_str.encode(), dtype=torch.uint8)
    # dist.broadcast(workspace, 0)  # Assuming rank 0 is the main process

    # workspace = workspace.decode()
        
    loss_str = 'loss'
    assert (opt.lambda_rendering + opt.lambda_splatter + opt.lambda_lpips > 0), 'Must have at least one loss'
    if opt.lambda_rendering > 0:
        loss_str+=f'_render{opt.lambda_rendering}'
    elif opt.lambda_alpha > 0:
        loss_str+=f'_alpha{opt.lambda_alpha}'
    if opt.lambda_splatter > 0:
        loss_str+=f'_splatter{opt.lambda_splatter}'
    if opt.lambda_lpips > 0:
        loss_str+=f'_lpips{opt.lambda_lpips}'
   
    desc = opt.desc
    if opt.vae_on_splatter_image:
        desc += "-vae_on_splatter_image"
    if opt.codes_from_encoder:
        desc += "-codes_from_encoder"
    elif opt.codes_from_diffusion:
        desc += "-codes_from_diffusion"
    elif opt.codes_from_cache:
        desc += "-codes_from_cache"
    
    
    assert (opt.one_step_diffusion is None) or (opt.lipschitz_mode is None)
    if opt.one_step_diffusion is not None:
        desc += f"_ONE_STEP_T={opt.one_step_diffusion}"
    if opt.lipschitz_mode is not None:
        desc += f"_lipschitz_mode={opt.lipschitz_mode}_coeff={opt.lipschitz_coefficient}"
        
        
    opt.workspace = os.path.join(opt.workspace, f"{time_str}-{desc}-{loss_str}-lr{opt.lr}-{opt.lr_scheduler}")
    
    if accelerator.is_main_process:
        assert not os.path.exists(opt.workspace)
        print(f"makdir: {opt.workspace}")
        os.makedirs(opt.workspace, exist_ok=True)
        writer = tensorboard.SummaryWriter(opt.workspace)
    
    # real_workspace = sorted(os.listdir(os.path.dirname(opt.workspace)))[-1]
    # opt.workspace = real_workspace
    print(f"workspace: {opt.workspace}")

    # # broadcast the opt.workspace to all processes
    # workspace_tensor = torch.tensor(list(opt.workspace.encode()), device="cuda", dtype=torch.uint8)
    # workspace_info = {'workspace tensor': workspace_tensor}
    # # Broadcast the workspace_info dictionary
    # workspace_info = broadcast(workspace_info, from_process=0)

    # # Decode the workspace string from the tensor
    # opt.workspace = bytes(workspace_info['workspace tensor'].tolist()).decode()

    # # Convert workspace string to a tensor
    # workspace_tensor = torch.tensor(bytearray(opt.workspace, 'utf-8'), dtype=torch.uint8).to("cuda")
    # # Broadcast the tensor
    # broadcasted_workspace_tensor = broadcast(workspace_tensor)
    # # Decode the tensor back to a string
    # decoded_workspace = broadcasted_workspace_tensor.cpu().numpy().tobytes().decode('utf-8')

    # # Use the decoded workspace
    # opt.workspace = decoded_workspace
    # print(f"Decoded workspace: {opt.workspace}")
    
    # accelerator.wait_for_everyone() 
    
    if opt.codes_from_cache:
        if opt.code_cache_dir is not None:
            code_cache_dir = opt.code_cache_dir
        else:
            resume_dir = os.path.dirname(opt.resume)
            if "epoch" in os.path.basename(resume_dir):
                resume_dir = os.path.dirname(resume_dir)
            code_cache_dir = os.path.join(resume_dir, "code_dir")
            assert os.isdir(code_cache_dir)
        print(f"Codes cache are loaded from:{code_cache_dir}")

    if accelerator.is_main_process:
        src_snapshot_folder = os.path.join(opt.workspace, 'src')
        ignore_func = lambda d, files: [f for f in files if f.endswith('__pycache__')]
        # for folder in ['core', 'scripts', 'zero123plus']:
        for folder in ['core', 'scripts']:
            dst_dir = os.path.join(src_snapshot_folder, folder)
            shutil.copytree(folder, dst_dir, ignore=ignore_func, dirs_exist_ok=True)
        for file in ['main_zero123plus_v4_batch_code_inference.py']:
            dest_file = os.path.join(src_snapshot_folder, file)
            shutil.copy2(file, dest_file)
        
    # resume
    resume = not opt.vae_on_splatter_image
    if resume:
        assert opt.resume is not None ## only for decoder
        print(f"Resume from ckpt: {opt.resume}")
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(opt.resume, device='cpu')
        else:
            ckpt = torch.load(opt.resume, map_location='cpu')
        
        # tolerant load (only load matching shapes)
        # model.load_state_dict(ckpt, strict=False)
        state_dict = model.state_dict()
        for k, v in ckpt.items():
            # if "lora" in k:
            #     print(f"not loading: {k}")
            #     # continue
            if k in state_dict: 
                if state_dict[k].shape == v.shape:
                    state_dict[k].copy_(v)
                else:
                    accelerator.print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
            else:
                accelerator.print(f'[WARN] unexpected param {k}: {v.shape}')
            
            # del state_dict[k]
        
        # non_loaded_params = [name for name, _ in state_dict.items() if 'lora' not in name]
        # print(non_loaded_params)
        # print("non_loaded_params that are not in ckpt")
        # decoder_params = [name for name, _ in ckpt.items() if 'decoder' in name]
        # print(decoder_params)
        # print("decoder_params")
        # st()
    
    # ## also load pretrained unet
    # if opt.resume_unet is not 
        

    # No need to copy the code_dir: handled by load_scenes already.
    train_dataset = Dataset(opt, training=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_dataset = Dataset(opt, training=False, prepare_white_bg=True)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )


    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.05, betas=(0.9, 0.95))

    # scheduler (per-iteration)
    if opt.lr_scheduler == 'CosAnn':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3000, eta_min=1e-6)
    elif opt.lr_scheduler == 'OneCyc':
        total_steps = opt.num_epochs * len(train_dataloader)
        pct_start = 3000 / total_steps
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.lr, total_steps=total_steps, pct_start=pct_start)
    elif opt.lr_scheduler == 'Plat':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=opt.lr_scheduler_factor, patience=opt.lr_scheduler_patience, verbose=True, min_lr=opt.min_lr_scheduled)
    else:
        assert ValueError('Not a valid lr_scheduler option.')

    # accelerate
    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, scheduler
    )
   
            # eval
    with torch.no_grad():
        model.eval()

        total_loss = 0
        total_psnr = 0
        total_loss_splatter = 0 #torch.tensor([0]).to()
        total_loss_rendering = 0 #torch.tensor([0])
        total_loss_alpha = 0
        total_loss_lpips = 0

        
        if opt.codes_from_diffusion or opt.vae_on_splatter_image:
                    
            # Load the pipeline
            pipeline_0123 = DiffusionPipeline.from_pretrained(
                "sudo-ai/zero123plus-v1.1", custom_pipeline="/mnt/kostas-graid/sw/envs/chenwang/workspace/diffgan/training/modules/zero123plus.py",
                torch_dtype=torch.float32
            )
            pipeline_0123.to('cuda:0')
            
            pipeline = model.pipe
            # pipeline = pipeline_0123
            # print("pipeline 0123")
            print(pipeline.unet) # check whether lora is here
            # st()

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
            
            output_path = f"{opt.workspace}/zero123plus/outputs_v3_inference_my_decoder"
            
            pipeline.prepare()
            guidance_scale = 4.0


        gt_attr_keys = ['pos', 'opacity', 'scale', 'rotation', 'rgbs']
        start_indices = [0, 3, 4, 7, 11]
        end_indices = [3, 4, 7, 11, 14]

        attr_map = {key: (si, ei) for key, si, ei in zip (gt_attr_keys, start_indices, end_indices)}

        group_scale = True
        if not group_scale:
            attr_map.update({
                'scale-x': (4,5),
                'scale-y': (5,6),
                'scale-z': (6,7),
                "z-depth": (2,3),
                "xy-offset": (0,2), # TODO: actually can directly take xyz during real training. Now for vis purpose only
            })

            attr_cannot_be_encoded = ["scale", "pos"]
            for key in attr_cannot_be_encoded:
                del attr_map[key]
        else:
            # attr_map.update({
                # 'scale-x': (4,5),
                # 'scale-y': (5,6),
                # 'scale-z': (6,7),
                # "z-depth": (2,3),
                # "xy-offset": (0,2), # TODO: actually can directly take xyz during real training. Now for vis purpose only
            # })

            # attr_cannot_be_encoded = ["pos"]
            # for key in attr_cannot_be_encoded:
            #     del attr_map[key]
            pass

        # assert # TODO: add check to cover each channel exactly once
        print(f"Please confirm this attr to encode:\n{attr_map}")
        # st()
        
        print(f"Save to run dir: {opt.workspace}")
        for i, data in enumerate(test_dataloader):
            if i > 40:
                exit(0)
            
                
            scene_name = data["scene_name"][0]
            if i < 5 or scene_name == "0a9b36d36e904aee8b51e978a7c0acfd":
                pass
            else:
                continue
            
            directory = f'{opt.workspace}/eval_ckpt/{accelerator.process_index}_{i}_{scene_name}'
            if not os.path.exists(directory):
                os.makedirs(directory)
            

            if opt.vae_on_splatter_image:

                print("---------- Begin vae_on_splatter_image ---------------")
                        
                path =f'/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/{scene_name}/000.png'
            
                name = path.split('/')[-2]
                name = f"{i}_{name}"
                os.makedirs(os.path.join(output_path, name), exist_ok=True)

                img = to_rgb_image(Image.open(path))
                
                img.save(os.path.join(output_path, f'{name}/cond.png'))
            

                with torch.no_grad():
            
                    # reshape splatter 
                    # make input 6 views into a 3x2 grid
                    splatters_mv = einops.rearrange(data["splatters_output"], 'b (h2 w2) c h w -> b c (h2 h) (w2 w)', h2=3, w2=2) 
                    # vae.encode

                    decoded_attr_image_dict = {}

                    for attr_to_encode, (start_i, end_i) in attr_map.items():

                       
                        splatter_attr = splatters_mv[:,start_i:end_i,...]
                        print(f"Attr {attr_to_encode}: min={splatter_attr.min()} max={splatter_attr.max()}")

                        # if attr_to_encode == "rotation":
                        # if True:
                        # if attr_to_encode not in ["rgbs", "opacity", "z-depth"]: # passed "xy-offset"
                        # if attr_to_encode not in ["xy-offset"]: # passed  , "z-depth"
                        # if attr_to_encode not in ["z-depth"]: # passed 
                        if attr_to_encode not in ["rotation"]: # passed 
                        # if  "scale" not in attr_to_encode: # passed
                            # print(f"Using attr : {attr_to_encode}")
                            decoded_attr_image_dict[attr_to_encode] =  splatter_attr # TODO: currently skip the encoding and decoding of rotation for simplicity
                            continue
                        else:
                            print(f"Diffusing attr : {attr_to_encode}")
                        
                        sp_min, sp_max = None, None

                        # process the channels
                        if end_i - start_i == 1:
                            print(f"repeat attr {attr_to_encode} for 3 times")
                            splatter_attr = splatter_attr.repeat(1, 3, 1, 1) # [0,1]
                        elif end_i - start_i == 3:
                            pass
                        elif attr_to_encode == "xy-offset":
                            # ## normalize to [0,1]
                            # sp_min, sp_max =  -1., 1.
                            # splatter_attr = (splatter_attr - sp_min) / (sp_max - sp_min)
                            ## cat one more dim
                            splatter_attr = torch.cat((splatter_attr, 0.5 * torch.ones_like(splatter_attr[:,0:1,...])), dim=1)
                        elif attr_to_encode == "rotation":
                            # st() # assert 4 is on the last dim
                            # quaternion to axis angle
                            quat = einops.rearrange(splatter_attr, 'b c h w -> b h w c')
                            axis_angle = quaternion_to_axis_angle(quat)
                            splatter_attr = einops.rearrange(axis_angle, 'b h w c -> b c h w')
                            # st()

                        else:
                            raise ValueError(f"The dimension of {attr_to_encode} is problematic to encode")
                        
                        if "scale" in attr_to_encode:
                            # use log scale
                            splatter_attr = torch.log(splatter_attr)
                            
                            print(f"{attr_to_encode} log min={splatter_attr.min()} max={splatter_attr.max()}")
                            sp_min, sp_max =  -10., -2.
                            splatter_attr = (splatter_attr - sp_min) / (sp_max - sp_min) # [0,1]
                            splatter_attr = splatter_attr.clip(0,1)

                        elif attr_to_encode in ["z-depth", "xy-offset", "pos"] :
                            # sp_min, sp_max =  splatter_attr.min(), splatter_attr.max()
                            # sp_min, sp_max =  -1., 1.
                            sp_min, sp_max =  -0.7, 0.7
                            splatter_attr = (splatter_attr - sp_min) / (sp_max - sp_min)
                            # splatter_attr = splatter_attr.clip(0,1) 
                        
                       
                        print(f"Normed attr {attr_to_encode}: min={splatter_attr.min()} max={splatter_attr.max()}")
                        

                        sp_image = splatter_attr * 2 - 1 # [map to range [-1,1]]
                        print(f"Normed attr [-1, 1] {attr_to_encode}: min={sp_image.min()} max={sp_image.max()}")
                        
                        # Save image before encoding
                        mv_image = einops.rearrange((sp_image[0].clip(-1,1)+1).cpu().numpy()*127.5, 'c h w-> h w c').astype(np.uint8) 
                        Image.fromarray(mv_image).save(os.path.join(output_path, f'{name}/{attr_to_encode}_to_encode.png'))
                        
                        # encode: splatter attr -> latent 
                        # sp_image_original = sp_image.clone()
                        sp_image = scale_image(sp_image)
                        sp_image = pipeline.vae.encode(sp_image).latent_dist.sample() * pipeline.vae.config.scaling_factor
                        latents = scale_latents(sp_image)

                        #  decode: latents -> platter attr
                        latents1 = unscale_latents(latents)
                        image = pipeline_0123.vae.decode(latents1 / pipeline.vae.config.scaling_factor, return_dict=False)[0]
                        image = unscale_image(image)


                        # save decoded image
                        mv_image_numpy = einops.rearrange((image[0].clip(-1,1)+1).cpu().numpy()*127.5, 'c h w-> h w c').astype(np.uint8) 
                        Image.fromarray(mv_image_numpy).save(os.path.join(output_path, f'{name}/{attr_to_encode}_pred.png'))

                        # scale back to original range
                        mv_image = image # [b c h w], in [-1,1]
                        
                        # if attr_to_encode in[ "z-depth", "xy-offset"]:
                        #     sp_image_o = mv_image
                        #     # st() # NOTE: try clip z-depth? No need, they are already within the range
                        # else:
                        sp_image_o = 0.5 * (mv_image + 1) # [map to range [0,1]]

                        print(f"Decoded attr [-1,1] {attr_to_encode}: min={mv_image.min()} max={mv_image.max()}")
                        print(f"Decoded attr [0,1] {attr_to_encode}: min={sp_image_o.min()} max={sp_image_o.max()}")

                        if "scale" in attr_to_encode:
                            # v2
                            sp_image_o = sp_image_o.clip(0,1) 
                            sp_image_o = sp_image_o * (sp_max - sp_min) + sp_min
                            
                            print(f"Decoded attr not clip [0,1] {attr_to_encode}: min={sp_image_o.min()} max={sp_image_o.max()}")
                            sp_image_o = torch.exp(sp_image_o)
                            print(sp_min, sp_max)
                            print(f"Decoded attr [unscaled] {attr_to_encode}: min={sp_image_o.min()} max={sp_image_o.max()}")

                        # elif attr_to_encode == "z-depth":
                            # sp_image_o = sp_image_o * (sp_max - sp_min) + sp_min
                        
                        elif attr_to_encode in[ "z-depth", "xy-offset", "pos"]:
                            sp_image_o = sp_image_o * (sp_max - sp_min) + sp_min

                        if attr_to_encode == "xy-offset": 
                            sp_image_o = sp_image_o[:,:2] # FIXME: ...,2??
                        
                        if attr_to_encode == "rotation": 
                          
                            ag = einops.rearrange(sp_image_o, 'b c h w -> b h w c')
                            quaternion = axis_angle_to_quaternion(ag)
                            sp_image_o = einops.rearrange(quaternion, 'b h w c -> b c h w')
                            # st()
                        
                        if end_i - start_i == 1:
                            # print(torch.allclose(torch.mean(sp_image_o, dim=1, keepdim=True), sp_image_o))
                            # st()
                            print(f"Decoded attr [unscaled, before mean] {attr_to_encode}: min={sp_image_o.min()} max={sp_image_o.max()}")
                            sp_image_o = torch.mean(sp_image_o, dim=1, keepdim=True) # avg.
                        
                            # sp_image_o = torch.median(sp_image_o, dim=1, keepdim=True).values # 
                            # sp_image_o = torch.max(sp_image_o, dim=1, keepdim=True).values # .
                            # st()
                            print(f"Decoded attr [unscaled, after median] {attr_to_encode}: min={sp_image_o.min()} max={sp_image_o.max()}")
                        
                        # save in the dict
                        decoded_attr_image_dict.update({attr_to_encode:sp_image_o})


                        print(f"Decoded attr [unscaled] {attr_to_encode}: min={sp_image_o.min()} max={sp_image_o.max()}")
                        # st()
                    # save gt 6 input views
                    gt_white_images = einops.rearrange(data['input'], 'b (h2 w2) c h w -> b c (h2 h) (w2 w)', h2=3, w2=2) 
                    gt_image = einops.rearrange((gt_white_images[0].clip(-1,1)).cpu().numpy()*255, 'c h w-> h w c').astype(np.uint8) 
                    Image.fromarray(gt_image).save(os.path.join(output_path, f'{name}/gt.png'))

                    ## render splatter 
                    render_splatter_images = True
                    if not render_splatter_images:
                        continue

                    # reshape to original splatter image shape for splatter rendering
                    ## cat all attrs
                    if not group_scale:
                        ordered_attr_list = ["xy-offset", "z-depth", # 0-3
                                            'opacity', # 3-4
                                            'scale-x', 'scale-y', 'scale-z', # 4-7
                                            "rotation", # 7-11
                                            "rgbs", # 11-14
                                            ] # must be an ordered list according to the channels
                    else:
                        ordered_attr_list = ["pos", # 0-3
                                            'opacity', # 3-4
                                            'scale', # 4-7
                                            "rotation", # 7-11
                                            "rgbs", # 11-14
                                            ] # must be an ordered list according to the channels
                    attr_image_list = [decoded_attr_image_dict[attr] for attr in ordered_attr_list ]
                    # [print(t.shape) for t in attr_image_list]
                    splatter_mv = torch.cat(attr_image_list, dim=1)

                    ## reshape 
                    splatters_to_render = einops.rearrange(splatter_mv, 'b c (h2 h) (w2 w) -> b (h2 w2) c h w', h2=3, w2=2) 

                    
                    # gs.render:
                    ## decoded image
                    gaussians = fuse_splatters(splatters_to_render)
                    bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device) * 0.5
                    gs_results = model.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
                    
                    # save gs.rendered images
                    # st()
                    pred_images = gs_results['image'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    kiui.write_image(os.path.join(output_path, f'{name}/gs_render_rgb.png'), pred_images)
        

                    pred_alphas = gs_results['alpha'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                    pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                    kiui.write_image(os.path.join(output_path, f'{name}/gs_render_alpha.png'), pred_alphas)

                    ## gt splatter image
                    gaussians = fuse_splatters(data["splatters_output"])
                    bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device) * 0.5
                    gs_results = model.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
                    
                    # save gs.rendered images
                    # st()
                    pred_images = gs_results['image'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    kiui.write_image(os.path.join(output_path, f'{name}/gs_render_rgb_gt.png'), pred_images)
        

                    pred_alphas = gs_results['alpha'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                    pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                    kiui.write_image(os.path.join(output_path, f'{name}/gs_render_alpha_gt.png'), pred_alphas)


                    # skip the remaining inference: trained decoder from codes
                    continue
                    


            elif opt.codes_from_diffusion:
                
                print("---------- Begin original inference ---------------")
                        
                path =f'/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/{scene_name}/000.png'
            
                name = path.split('/')[-2]
               
                inference_on_unseen = True
                if inference_on_unseen: 
                    img = to_rgb_image(Image.open(requests.get("https://d.skis.ltd/nrp/sample-data/lysol.png", stream=True).raw))
                    name = "lysol"
                else:
                    img = to_rgb_image(Image.open(path)) 
                
                
                name = f"{i}_{name}"
                os.makedirs(os.path.join(output_path, name), exist_ok=True)

                img.save(os.path.join(output_path, f'{name}/cond.png'))
                cond = [img]
                print(img)
                
                prompt_embeds, cak = pipeline.prepare_conditions(cond, guidance_scale=4.0)
                print(f"cak: {cak['cond_lat'].shape}") # always 64x64, not affected by cond size
                pipeline.scheduler.set_timesteps(75, device='cuda:0')
                # if opt.one_step_diffusion is not None:
                #     pipeline.scheduler.set_timesteps(opt.one_step_diffusion, device='cuda:0')
                    
                timesteps = pipeline.scheduler.timesteps
            
                latents  = torch.randn([1, pipeline.unet.config.in_channels, 120, 80], device='cuda:0', dtype=torch.float32)
                latents_init = latents.clone().detach()

                with torch.no_grad():
                    # if opt.one_step_diffusion is not None:
                       
                    #     text_embeddings = prompt_embeds
                    #     t = torch.tensor([opt.one_step_diffusion]).to(timesteps.device)
                    #     print("ONE STEP DIFFUSION timestep =",t)
                    #     # st()
                    #     x = model.predict_x0(
                    #         latents, text_embeddings, t=t, guidance_scale=guidance_scale, 
                    #         cross_attention_kwargs=cak, scheduler=pipeline.scheduler, model='zero123plus')
                    #     latents = x
                        
                    #     timesteps = [] # skip the step-by-step inference

                    
                    lipschitz_analysis_zero123p = False

                    if opt.lipschitz_mode is not None and lipschitz_analysis_zero123p:
                        # get gt latents from encoder
                        # make input 6 views into a 3x2 grid
                        images = einops.rearrange(data['input'], 'b (h2 w2) c h w -> b c (h2 h) (w2 w)', h2=3, w2=2) 
                        latents = model.encode_image(images) # [B, self.pipe.unet.config.in_channels, 120, 80]

                        # add noise 
                        if opt.lipschitz_mode == "gaussian_noise":
                            noise = torch.randn_like(latents, device=latents.device)
                        elif opt.lipschitz_mode == "constant":
                            print(f"Adding constant lipschitz noise of scale {opt.lipschitz_coefficient}")
                            noise = torch.ones_like(latents, device=latents.device)
                        else:
                            raise ValueError ("invalid mode type for lipschitz analysis")

                        codes_gt = latents.clone()
                        latents += noise * opt.lipschitz_coefficient
                        latent_loss = F.mse_loss(latents, codes_gt)
                        print(f"latent loss = {latent_loss}") 

                        timesteps = [] # skip the step-by-step inference
                    
                    encode_splatter = False
                    if encode_splatter:

                        # reshape splatter 
                         # make input 6 views into a 3x2 grid
                        splatters_mv = einops.rearrange(data["splatters_output"], 'b (h2 w2) c h w -> b c (h2 h) (w2 w)', h2=3, w2=2) 
                        # vae.encode

                        ## RGB
                        # splatter_rgb = splatters_mv[:,11:14,...]
                        # image = splatter_rgb * 2 - 1 # [map to range [-1,1]]
                       

                        # # Opacity
                        # splatter_opacity = splatters_mv[:,3:4,...].repeat(1, 3, 1, 1) # [0,1]
                        # image = splatter_opacity * 2 - 1 # [map to range [-1,1]]

                        # gt_attr_keys = ['pos', 'opacity', 'scale', 'rotation', 'rgbs']
                        # start_indices = [0, 3, 4, 7, 11]
                        # end_indices = [3, 4, 7, 11, 14]
                        
                        attr_to_encode = "z-depth"
                        start_i, end_i = attr_map[attr_to_encode]          

                        splatter_attr = splatters_mv[:,start_i:end_i,...]
                        if end_i - start_i == 1:
                            splatter_attr = splatter_attr.repeat(1, 3, 1, 1) # [0,1]
                        elif end_i - start_i == 3:
                            pass
                        elif attr_to_encode == "xy-offset":
                            # st()
                            ## normalize to [0,1]
                            splatter_attr = (splatter_attr - splatter_attr.min()) / (splatter_attr.max() - splatter_attr.min())
                            ## cat one more dim
                            splatter_attr = torch.cat((splatter_attr, 0.5 * torch.ones_like(splatter_attr[:,0:1,...])), dim=1)
                        else:
                            raise ValueError(f"The dimension of {attr_to_encode} is problematic to encode")
                        
                        print(f"Attr {attr_to_encode}: min={splatter_attr.min()} max={splatter_attr.max()}")
                        if "scale" in attr_to_encode:
                            splatter_attr *= 20
                            splatter_attr = splatter_attr.clip(0,1)
                            print(f"New range of {attr_to_encode}: min={splatter_attr.min()} max={splatter_attr.max()}")
                        elif attr_to_encode == "z-depth":
                            splatter_attr = (splatter_attr - splatter_attr.min()) / (splatter_attr.max() - splatter_attr.min())
                            # splatter_attr = splatter_attr.clip(0,1)
                           
                        # st()

                        sp_image = splatter_attr * 2 - 1 # [map to range [-1,1]]
                        
                        # Save image
                        mv_image = einops.rearrange((sp_image[0].clip(-1,1)+1).cpu().numpy()*127.5, 'c h w-> h w c').astype(np.uint8) 
                        Image.fromarray(mv_image).save(os.path.join(output_path, f'{name}/{attr_to_encode}_to_encode.png'))
                        
                        # encode
                        sp_image = scale_image(sp_image)
                        sp_image = pipeline.vae.encode(sp_image).latent_dist.sample() * pipeline.vae.config.scaling_factor
                        latents = scale_latents(sp_image)

                        timesteps = [] # skip the step-by-step inference
                    
                    for i, t in enumerate(timesteps):
                        print(f"enumerate(timesteps) t={t}")
                        # st()
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

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                    

                    # --------- 
                    print("Codes from diffusion (moved)")
                    data['codes'] = latents # torch.Size([1, 4, 120, 80])
                    print(f"code-diffusion: max={latents.max()} min={latents.min()} mean={latents.mean()}")
            
                    #####  # check latents
                    latents1 = unscale_latents(latents)
                   
                    # ### lgm deocder 
                    # z = latents1 / model.decoder.vae.config.scaling_factor
                    
                    # ud = model.decoder
                    # sample = ud.vae.post_quant_conv(z)
                    # latent_embeds = None
                    # sample = ud.decoder.conv_in(sample)
                    # upscale_dtype = next(iter(ud.decoder.up_blocks.parameters())).dtype
                    # sample = ud.decoder.mid_block(sample, latent_embeds)
                    # sample = sample.to(upscale_dtype)
                    # # up
                    # for i, up_block in enumerate(ud.decoder.up_blocks):
                    #     # print(f"{i}th upblock input: {sample.shape}")
                    #     sample = up_block(sample, latent_embeds)
                    
                    # # print(f"{i}th upblock output: {sample.shape}")
                    # # st()
                    
                    # sample = ud.decoder.conv_norm_out(sample) 
                    # sample = ud.decoder.conv_act(sample)
                    # image = ud.decoder.conv_out(sample)
                    
                    # ### --- [end] ---
                    
                    image = pipeline_0123.vae.decode(latents1 / pipeline.vae.config.scaling_factor, return_dict=False)[0]
                    image = unscale_image(image)

                    latents_init1 = unscale_latents(latents_init)
                    image_init = pipeline.vae.decode(latents_init1 / pipeline.vae.config.scaling_factor, return_dict=False)[0]
                    image_init = unscale_image(image_init)

                save_single_frames = False
                
                if save_single_frames:
                    mv_image = einops.rearrange((image[0].clip(-1,1)+1).cpu().numpy()*127.5, 'c (h2 h) (w2 w)-> (h2 w2) h w c', h2=3, w2=2).astype(np.uint8) 
                    for i, image in enumerate(mv_image):
                        image = rembg.remove(image).astype(np.float32) / 255.0
                        if image.shape[-1] == 4:
                            image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
                        Image.fromarray((image * 255).astype(np.uint8)).save(os.path.join(output_path, f'{name}/{i:03d}.png'))
                else:
                    mv_image = einops.rearrange((image[0].clip(-1,1)+1).cpu().numpy()*127.5, 'c h w-> h w c').astype(np.uint8) 
                    image = mv_image

                    white_bg = False
                    if white_bg:
                        image = rembg.remove(image).astype(np.float32) / 255.0
                
                        if image.shape[-1] == 4:
                            alpha_image = np.repeat((1 - image[..., 3:4]), repeats=3, axis=-1) # .astype(np.uint8).astype(np.float32)
                            # image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
                            image = image[..., :3] *(1 - alpha_image) + alpha_image
                    
                            Image.fromarray((alpha_image * 255).astype(np.uint8)).save(os.path.join(output_path, f'{name}/alpha.png'))
                        Image.fromarray((image * 255).astype(np.uint8)).save(os.path.join(output_path, f'{name}/pred.png'))

                        gt_white_images = einops.rearrange(data['input_white'], 'b (h2 w2) c h w -> b c (h2 h) (w2 w)', h2=3, w2=2) 
                        gt_image = einops.rearrange((gt_white_images[0].clip(-1,1)).cpu().numpy()*255, 'c h w-> h w c').astype(np.uint8) 
                        Image.fromarray(gt_image).save(os.path.join(output_path, f'{name}/gt_white.png'))
                    else:
                        Image.fromarray(image).save(os.path.join(output_path, f'{name}/pred.png'))

                        gt_white_images = einops.rearrange(data['input'], 'b (h2 w2) c h w -> b c (h2 h) (w2 w)', h2=3, w2=2) 
                        gt_image = einops.rearrange((gt_white_images[0].clip(-1,1)).cpu().numpy()*255, 'c h w-> h w c').astype(np.uint8) 
                        Image.fromarray(gt_image).save(os.path.join(output_path, f'{name}/gt.png'))


                print("---------- the above is original inference ---------------")
                
                # continue
                
                # debug_latent = True
                # if debug_latent:
                #     # Load the pipeline #TODO: this is only debug
                #     pipeline = DiffusionPipeline.from_pretrained(
                #         "sudo-ai/zero123plus-v1.1", custom_pipeline="/mnt/kostas-graid/sw/envs/chenwang/workspace/diffgan/training/modules/zero123plus.py",
                #         torch_dtype=torch.float32
                #     )
                #     pipeline.to('cuda:0')
                    
                # else:
                #     pipeline = model.pipe
                
                # pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                #         pipeline.scheduler.config, timestep_spacing='trailing'
                #     )
                # # TODO: try the default scheduler
                # pipeline.prepare()
                
                # guidance_scale = 4.0
                # cond_path=f'/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/{scene_name}/000.png'
                # img = to_rgb_image(Image.open(cond_path))
                    
                # img.save(f'{directory}/cond.jpg')
                # cond = [img]

                # prompt_embeds, cak = pipeline.prepare_conditions(cond, guidance_scale=4.0)
                # print(f"cak: {cak['cond_lat'].shape}") # always 64x64, not affected by cond size
                # model.pipe.scheduler.set_timesteps(75, device="cuda:0")
                # timesteps = model.pipe.scheduler.timesteps.to(torch.int64)
                # #TODO: check the timesteps every iter
                # print(timesteps)
                # # st()

                # latents = torch.randn([1, model.pipe.unet.config.in_channels, 120, 80], device="cuda:0", dtype=torch.float32)
                # latents_init = latents.clone().detach()
            
                # with torch.no_grad():
                    
                #     for _, t in enumerate(timesteps):
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
                #         latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0] # torch.Size([1, 4, 120, 80])
                    
                #     print("Codes from diffusion")
                #     data['codes'] = latents # torch.Size([1, 4, 120, 80])
                    
                #     if debug_latent:
                #         #####  # check latents
                #         latents1 = unscale_latents(latents)
                #         image = pipeline.vae.decode(latents1 / pipeline.vae.config.scaling_factor, return_dict=False)[0]
                #         image = unscale_image(image)
                        
                #         mv_image = einops.rearrange((image[0].clip(-1,1)+1).cpu().numpy()*127.5, 'c (h2 h) (w2 w)-> (h2 w2) h w c', h2=3, w2=2).astype(np.uint8) 
                #         for j, image in enumerate(mv_image):
                #             image = rembg.remove(image).astype(np.float32) / 255.0
                #             if image.shape[-1] == 4:
                #                 image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
                            
                #             Image.fromarray((image * 255).astype(np.uint8)).save(f'{directory}/{j:03d}.png')
        
            # elif not opt.codes_from_encoder:
            elif opt.codes_from_cache: # NOTE: make this more explicit
                ## ---- load or init code here ----
                
                if num_gpus==1:
                    codes = model.load_scenes(code_cache_dir, data, eval_mode=True)
                else:
                    codes = model.module.load_scenes(code_cache_dir, data, eval_mode=True)
                
                assert not (opt.one_step_diffusion is not None) and (opt.lipschitz_mode is not None)

                if opt.lipschitz_mode is not None:
                    if opt.lipschitz_mode == "gaussian_noise":
                        noise = torch.randn_like(codes, device=codes.device)
                    elif opt.lipschitz_mode == "constant":
                        print(f"Adding constant lipschitz noise of scale {opt.lipschitz_coefficient}")
                        noise = torch.ones_like(codes, device=codes.device)
                    else:
                        raise ValueError ("invalid mode type for lipschitz analysis")

                    # num_levels = 100
                    # disturb_level = torch.linspace(num_levels)/num_levels
                    codes_gt = codes.clone()
                    codes += noise * opt.lipschitz_coefficient
                    latent_loss = F.mse_loss(codes, codes_gt)
                    print(f"latent loss = {latent_loss}")
                    
                    

                if opt.one_step_diffusion is not None: 
                    t = torch.tensor([opt.one_step_diffusion]).to(codes.device)
                    
                    noise = torch.randn_like(codes, device=codes.device)
                    noisy_latents = model.pipe.scheduler.add_noise(codes, noise, t)

                    # get cond embed
                    path =f'/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/{scene_name}/000.png'
                    output_path = f"{opt.workspace}/zero123plus/codes_from_cache"
                    name = path.split('/')[-2]
                    os.makedirs(os.path.join(output_path, name), exist_ok=True)
                    # model.pipe.prepare()
                    guidance_scale = 4.0
                    img = to_rgb_image(Image.open(path))
                    img.save(os.path.join(output_path, f'{name}/cond.png'))
                    cond = [img]
                    print(img)
                    text_embeddings, cak = model.pipe.prepare_conditions(cond, guidance_scale=4.0)
                    # -------
           
                    print("ONE STEP DIFFUSION timestep =",t)
                    # st()
                    codes = model.predict_x0(
                        noisy_latents, text_embeddings, t=t, guidance_scale=guidance_scale, 
                        cross_attention_kwargs=cak, scheduler=model.pipe.scheduler, model='zero123plus')
                   
                
                
                data['codes'] = codes # torch.Size([1, 4, 120, 80])
                # st()
                
                print(f"code-optimized: max={codes.max()} min={codes.min()} mean={codes.mean()}")
                
                # ---- finish code init ----
            elif opt.codes_from_encoder:
                print("codes_from_encoder, are you sure?")
                # codes = model.encode_image(data['input'])
                # print(f"code-encoder: max={codes.max()} min={codes.min()} mean={codes.mean()}")

            else:
                raise ValueError("Not a valid source of latent")
            
            out = model(data)

            psnr = out['psnr']
            total_psnr += psnr.detach()
            loss = out['loss']
            total_loss += loss.detach()
            if 'loss_splatter' in out.keys():
                total_loss_splatter += out['loss_splatter'].detach()
            if 'loss_rendering' in out.keys():
                total_loss_rendering += out['loss_rendering'].detach()
            elif 'loss_alpha' in out.keys():
                total_loss_alpha += out["loss_alpha"].detach()
            if 'loss_lpips' in out.keys():
                total_loss_lpips += out['loss_lpips'].detach()
    
            
            # save some images
            # if accelerator.is_main_process:
            if True:

                gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                kiui.write_image(f'{directory}/image_gt.jpg', gt_images)

                pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                kiui.write_image(f'{directory}/image_pred.jpg', pred_images)

                pred_alphas = out['alphas_pred'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                kiui.write_image(f'{directory}/image_alpha.jpg', pred_alphas)
                
                ## save white images
                pred_images_white = pred_images * pred_alphas + 1 * (1 - pred_alphas)
                kiui.write_image(f'{directory}/image_pred_white.jpg', pred_images_white)

                gt_images_white = data['images_output_white'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                gt_images_white = gt_images_white.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images_white.shape[1] * gt_images_white.shape[3], 3) # [B*output_size, V*output_size, 3]
                kiui.write_image(f'{directory}/image_gt_white.jpg', gt_images_white)
                
        
                # # add write images for splatter to optimize
                # pred_images = out['images_opt'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                # pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                # kiui.write_image(f'{opt.workspace}/eval_ckpt/{i}_image_splatter_opt.jpg', pred_images)

                # pred_alphas = out['alphas_opt'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                # pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                # kiui.write_image(f'{opt.workspace}/eval_ckpt/{i}_image_splatter_opt_alpha.jpg', pred_alphas)

                if len(opt.plot_attribute_histgram) > 0:
                    for splatters_pred_key in ['splatters_from_code']:
                        if splatters_pred_key == 'splatters_from_code':
                            splatters = out[splatters_pred_key]
                        else:
                            raise NotImplementedError
                        
                        gaussians = fuse_splatters(splatters)
                        gt_gaussians = fuse_splatters(data['splatters_output'])
                        
                        color_pairs = [('pink', 'teal'), ("red", "green"), ("orange", "blue"), ('purple', 'yellow'), ('cyan', 'brown')]

                        attr_map = {key: (si, ei, color_pair) for key, si, ei, color_pair in zip (gt_attr_keys, start_indices, end_indices, color_pairs)}
        
                        for attr in opt.plot_attribute_histgram:
                        
                            start_i, end_i, (gt_color, pred_color) = attr_map[attr]
                            # if opt.verbose_main:
                            #     print(f"plot {attr} in dim ({start_i}, {end_i})")
                            
                            gt_attr_flatten =  gt_gaussians[..., start_i:end_i] # [B, L, C]
                            pred_attr_flatten = gaussians[..., start_i:end_i]
                            
                            if attr in ['scale', 'opacity']:
                                gt_attr_flatten = torch.log(gt_attr_flatten).permute(0,2,1) # [B, C, L]
                                pred_attr_flatten = torch.log(pred_attr_flatten).permute(0,2,1) 
                                gt_attr_flatten = gt_attr_flatten.flatten().detach().cpu().numpy()
                                pred_attr_flatten = pred_attr_flatten.flatten().detach().cpu().numpy()

                                
                            else:
                                ## cannot flatten due to their meaning
                                print(f"not support the plotting of __{attr}__ yet")
                                continue
                            
                            # Manually define bin edges
                            bin_edges = np.linspace(min(min(gt_attr_flatten), min(pred_attr_flatten)), max(max(gt_attr_flatten), max(pred_attr_flatten)), num=50)

                            plt.hist(gt_attr_flatten, bins=bin_edges, color=gt_color, alpha=0.7, label=f'{attr}_gt')
                            plt.hist(pred_attr_flatten, bins=bin_edges, color=pred_color, alpha=0.3, label=f'{attr}_pred')
                            
                            # Add labels and legend
                            plt.xlabel('Value')
                            plt.ylabel('Frequency')
                            plt.legend()

                            # Save the plot as an image file (e.g., PNG)
                            name = f'histogram_epoch_batch{i}_{splatters_pred_key}_{attr}'
                            if attr == "scale":
                                name += f"_{opt.scale_act}_bias{opt.scale_act_bias}"
                            
                            if opt.normalize_scale_using_gt:
                                name += "normed_on_gt"
                                
                            plt.title(f'{name}')
                            plt.savefig(f'{opt.workspace}/eval_ckpt/{i}_{name}.jpg')
                        
                            # Clear the figure
                            plt.clf()
                    
    
        torch.cuda.empty_cache()

        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        if accelerator.is_main_process:
            total_psnr /= len(test_dataloader)
            # accelerator.print(f"[eval] epoch: {epoch} psnr: {psnr:.4f}")
            total_loss /= len(test_dataloader)
            total_loss_splatter /= len(test_dataloader)
            total_loss_rendering /= len(test_dataloader)
            total_loss_alpha /= len(test_dataloader)
            total_loss_lpips /= len(test_dataloader)
            
            accelerator.print(f"[eval] ckpt loss: {total_loss.item():.6f} psnr: {total_psnr.item():.4f} splatter_loss: {total_loss_splatter:.4f} rendering_loss: {total_loss_rendering:.4f} alpha_loss: {total_loss_alpha:.4f} lpips_loss: {total_loss_lpips:.4f} ")
           

if __name__ == "__main__":
    
    ### Ignore the FutureWarning from pipeline_stable_diffusion.py
    warnings.filterwarnings("ignore", category=FutureWarning, module="pipeline_stable_diffusion")
    
    main()
    
    # Reset the warning filter to its default state (optional)
    warnings.resetwarnings()
