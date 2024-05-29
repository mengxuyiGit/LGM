import tyro
import time
import random

import torch
from core.options import AllConfigs
from core.models_zero123plus_marigold_unet_rendering_loss_cross_domain import Zero123PlusGaussianMarigoldUnetCrossDomain, fuse_splatters
from core.dataset_v5_marigold import gt_attr_keys, start_indices, end_indices
from core.dataset_v5_marigold import ObjaverseDataset as Dataset

from accelerate import Accelerator, DistributedDataParallelKwargs
from safetensors.torch import load_file

import kiui
from datetime import datetime
import torch.utils.tensorboard as tensorboard
import shutil, os

from ipdb import set_trace as st
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
import torch.nn.functional as F

from kiui.cam import orbit_camera
import imageio

import warnings
from accelerate.utils import broadcast
import re
import multiprocessing as mp
import math
from accelerate.logging import get_logger
logger = get_logger(__name__, log_level="INFO")

from utils.format_helper import get_workspace_name
from utils.io_helper import print_grad_status

def store_initial_weights(model):
    """Stores the initial weights of the model for later comparison."""
    initial_weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            initial_weights[name] = param.data.clone()
    return initial_weights

def compare_weights(initial_weights, model):
    """Compares the initial weights to the current weights to check for updates."""
    updated = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Check if the current parameter is different from the initial
            if not torch.equal(initial_weights[name], param.data):
                print(f"Weight updated: {name}")
                updated = True
    if not updated:
        print("No weights were updated.")
        

# Set the start method to 'spawn'
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

def main(): 
    import sys
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
  
    # Introduce a delay based on the process rank to stagger the loading
    print(f"Sleep process {accelerator.process_index} before loading")
    time.sleep(accelerator.process_index * 5)  # Delay by 5 seconds per process index

    # Now load the model
    assert opt.model_type == "Zero123PlusGaussianMarigoldUnetCrossDomain", "Invalid model type"
    model =  Zero123PlusGaussianMarigoldUnetCrossDomain(opt)
    print(f"Model loaded by process {accelerator.process_index}")
    accelerator.wait_for_everyone()
    
    # Create workspace
    ## check the number of GPUs
    num_gpus = accelerator.num_processes
    if accelerator.is_main_process:
        print(f"Num gpus: {num_gpus}")
    ## create time-ordered prefix
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
    ## create folder
    opt.workspace = get_workspace_name(opt, time_str, num_gpus)
    if accelerator.is_main_process:
        assert not os.path.exists(opt.workspace)
        print(f"makdir: {opt.workspace}")
        os.makedirs(opt.workspace, exist_ok=True)
        writer = tensorboard.SummaryWriter(opt.workspace)
    print(f"workspace: {opt.workspace}")

    if accelerator.is_main_process:
        src_snapshot_folder = os.path.join(opt.workspace, 'src')
        ignore_func = lambda d, files: [f for f in files if f.endswith('__pycache__')]
        for folder in ['core', 'scripts', 'zero123plus']:
            dst_dir = os.path.join(src_snapshot_folder, folder)
            shutil.copytree(folder, dst_dir, ignore=ignore_func, dirs_exist_ok=True)
        for file in ['main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared_inference.py']:
            dest_file = os.path.join(src_snapshot_folder, file)
            shutil.copy2(file, dest_file)
        
    assert (opt.resume_decoder is not None) or (opt.resume_unet is not None)
    if opt.resume is not None:
        print(f"Resume from ckpt: {opt.resume}")
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(opt.resume, device='cpu')
        else:
            ckpt = torch.load(opt.resume, map_location='cpu')
        
        # tolerant load (only load matching shapes)
        # model.load_state_dict(ckpt, strict=False)
        state_dict = model.state_dict()
        for k, v in ckpt.items():
            if k in state_dict: 
                if state_dict[k].shape == v.shape:
                    # print("... copying ", k)
                    state_dict[k].copy_(v)
                else:
                    accelerator.print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
            else:
                accelerator.print(f'[WARN] unexpected param {k}: {v.shape}')
    
    # we allow resume from both decoder and unet
    if opt.resume_decoder is not None:
        print(f"Resume from decoder ckpt: {opt.resume_decoder}")
        if opt.resume_decoder.endswith('safetensors'):
            ckpt = load_file(opt.resume_decoder, device='cpu')
        else:
            ckpt = torch.load(opt.resume_decoder, map_location='cpu')
        
        # Prepare a set of parpameters that requires_grad=True in decoder
        trainable_decoder_params = set(f"vae.decoder.{name}" for name, para in model.vae.decoder.named_parameters())
        # checked: this set is equal to check with model.vae.decoder.named_parameters(), whether dupplicate names
        
        state_dict = model.state_dict()
        for k, v in ckpt.items():
            if k in trainable_decoder_params:
                if k in state_dict and state_dict[k].shape == v.shape:
                    print(f"Copying {k}")
                    state_dict[k].copy_(v)
                else:
                    if k not in state_dict:
                        accelerator.print(f'[WARN] Parameter {k} not found in model.')
                    else:
                        accelerator.print(f'[WARN] Mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
        
        print("Finish loading finetuned decoder.")
    
    if opt.resume_unet is not None:
        print(f"Resume from unet ckpt: {opt.resume_unet}")
        if opt.resume_unet.endswith("safetensors"):
            ckpt = load_file(opt.resume_unet, device="cpu")
        else:
            ckpt = torch.load(opt.resume_unet, device="cpu")
        
        # Prepare unet parameter list
        if opt.only_train_attention:
            trained_unet_parameters = set(f"unet.{name}" for name, para in model.unet.named_parameters() if "transformer_blocks" in name)
        else:
            trained_unet_parameters = set(f"unet.{name}" for name, para in model.unet.named_parameters())
        
        state_dict = model.state_dict()
        for k, v in ckpt.items():
            if k in trained_unet_parameters:
                print(f"Copying {k}")
                state_dict[k].copy_(v)
            else:
                if k not in state_dict:
                    accelerator.print(f"[WARN] Parameter {k} not found in model. ")
                elif v.shape == state_dict[k].shape:
                    assert opt.only_train_attention
                else:
                    accelerator.print(f"[WARN] Mismatchinng shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.")
                
        print("Finish loading trained unet.")
    
    torch.cuda.empty_cache()
    
    train_dataset = Dataset(opt, training=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_dataset = Dataset(opt, training=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    

    # optimizer
    assert not (opt.finetune_decoder or opt.train_unet)
    
    model, test_dataloader = accelerator.prepare(
        model, test_dataloader 
    )

    # loop
    # with torch.autograd.profiler.profile(use_cuda=True, record_shapes=True, profile_memory=True) as prof
    with torch.no_grad():  
        model.eval()
        num_samples_eval = 100
        total_psnr = 0
        total_psnr_LGM = 0
        total_lpips = 0
        total_lpips_LGM = 0
        
        if opt.log_each_attribute_loss or (opt.train_unet_single_attr is not None):
            from core.dataset_v5_marigold import ordered_attr_list
            if opt.train_unet_single_attr is not None:
                ordered_attr_list = opt.train_unet_single_attr 
                
            total_attr_loss_dict = {}
            for _attr in ordered_attr_list:
                total_attr_loss_dict[f"loss_{_attr}"] = 0

        with open(f"{opt.workspace}/metrics.txt", "w") as f:
            print(f"Total samples to eval = {num_samples_eval}", file=f)
        
        for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), disable=(opt.verbose_main)):
            if i == num_samples_eval:
                break
        
            out = model(data, save_path=f'{opt.workspace}/eval_inference', prefix=f"{accelerator.process_index}_{i}_")
    
            if opt.train_unet_single_attr is not None:
                for _attr in ordered_attr_list:
                    total_attr_loss_dict[f"loss_{_attr}"] += out[f"loss_{_attr}"].detach()
                
                    
            else:
                loss = out['loss']
                loss_latent = out['loss_latent'] if 'loss_latent' in out.keys() else torch.zeros_like(loss)
                
                lpips = out['loss_lpips']
                lpips_LGM = out['loss_lpips_LGM']
                
                psnr = out['psnr']
                psnr_LGM = out['psnr_LGM']
                
                if psnr > 50:
                    # not count this sample
                    num_samples_eval -= 1
                else:
                    total_psnr += psnr.detach()
                    total_psnr_LGM += psnr_LGM.detach()
                    total_lpips += lpips
                    total_lpips_LGM += lpips_LGM
                    if opt.log_each_attribute_loss:
                        for _attr in ordered_attr_list:
                            total_attr_loss_dict[f"loss_{_attr}"] += out[f"loss_{_attr}"].detach()
    
           
            # save some images
            # if True:
            if opt.train_unet_single_attr is None:
                # gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                # gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                # # kiui.write_image(f'{opt.workspace}/eval_epoch_{epoch}/{accelerator.process_index}_{i}_image_gt.jpg', gt_images)
                # kiui.write_image(f'{opt.workspace}/eval_inference/{accelerator.process_index}_{i}_image_gt.jpg', gt_images)

                # pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                # pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                # # kiui.write_image(f'{opt.workspace}/eval_epoch_{epoch}/{accelerator.process_index}_{i}_image_pred.jpg', pred_images)
                # kiui.write_image(f'{opt.workspace}/eval_inference/{accelerator.process_index}_{i}_image_pred.jpg', pred_images)

                # pred_alphas = out['alphas_pred'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                # pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                # # kiui.write_image(f'{opt.workspace}/eval_epoch_{epoch}/{accelerator.process_index}_{i}_image_alpha.jpg', pred_alphas)
                # kiui.write_image(f'{opt.workspace}/eval_inference/{accelerator.process_index}_{i}_image_alpha.jpg', pred_alphas)
                
                # # also render the LGM inferenced splatter 
                # pred_images = out['images_pred_LGM'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                # pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                # # kiui.write_image(f'{opt.workspace}/eval_epoch_{epoch}/{accelerator.process_index}_{i}_image_pred.jpg', pred_images)
                # kiui.write_image(f'{opt.workspace}/eval_inference/{accelerator.process_index}_{i}_image_pred_LGM.jpg', pred_images)

                # pred_alphas = out['alphas_pred_LGM'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                # pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                # # kiui.write_image(f'{opt.workspace}/eval_epoch_{epoch}/{accelerator.process_index}_{i}_image_alpha.jpg', pred_alphas)
                # kiui.write_image(f'{opt.workspace}/eval_inference/{accelerator.process_index}_{i}_image_alpha_LGM.jpg', pred_alphas)

                
                # 5-in-1
                five_in_one = torch.cat([data['images_output'], out['images_pred_LGM'], out['alphas_pred_LGM'].repeat(1,1,3,1,1), out['images_pred'], out['alphas_pred'].repeat(1,1,3,1,1)], dim=0)
                gt_images = five_in_one.detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                kiui.write_image(f'{opt.workspace}/eval_inference/{accelerator.process_index}_{i}_Ugt_Mlgm_Dpred.jpg', gt_images)

                # render 360 video 
                if opt.fancy_video or opt.render_video:
                    
                    device = data['images_output'].device
                    
                    tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
                    proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
                    proj_matrix[0, 0] = 1 / tan_half_fov
                    proj_matrix[1, 1] = 1 / tan_half_fov
                    proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
                    proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
                    proj_matrix[2, 3] = 1
                    
                    images_dict = {}
                    for key in ["gaussians_LGM", "gaussians_pred"]:
                        gaussians = out[key]
                        images = []
                        elevation = 0
                        

                        if opt.fancy_video:

                            azimuth = np.arange(0, 720, 4, dtype=np.int32)
                            for azi in tqdm(azimuth):
                                
                                cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

                                cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                                
                                # cameras needed by gaussian rasterizer
                                cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                                cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                                cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                                
                                scale = min(azi / 360, 1)
                                image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=scale)['image']

                                
                                images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))
                        elif opt.render_video:
                            azimuth = np.arange(0, 360, 2, dtype=np.int32)
                            for azi in tqdm(azimuth):
                                
                                cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

                                cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                                
                                # cameras needed by gaussian rasterizer
                                cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                                cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                                cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                                image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
                                # additional video to vis pts pos
                                image_pos = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=4/360)['image']
                                image = torch.cat([image, image_pos], dim=-1)
                                
                                images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))

                        images_dict[key] = np.concatenate(images, axis=0)
                        # images = np.concatenate(images, axis=0)
                        # imageio.mimwrite(f'{opt.workspace}/eval_inference/{accelerator.process_index}_{i}_video_{key.replace("gaussians_", "")}.mp4', images, fps=30)
                    
                    # save all four videos in a row
                    images = np.concatenate([images_dict[key] for key in ["gaussians_LGM", "gaussians_pred"]], axis=2) # cat on width
                    imageio.mimwrite(f'{opt.workspace}/eval_inference/{accelerator.process_index}_{i}_video_LGM_pred.mp4', images, fps=30)
                
                
                with open(f"{opt.workspace}/metrics.txt", "a") as f:
                     
                    if psnr > 50:
                        print("------[invalid]------", file=f)
            
                    our_loss_str = f"{i} - our_psnr: {psnr:.3f} \t lpips: {lpips:.3f}"
                    LGM_loss_str = f"{i} - LGM_psnr: {psnr_LGM:.3f} \t lpips: {lpips_LGM:.3f}"
                    if opt.log_each_attribute_loss:
                        for _attr in ordered_attr_list:
                            _loss_attr = out[f'loss_{_attr}'].item()
                            our_loss_str += f" \t {_attr}: {_loss_attr:.3f}"
                   
                    print(our_loss_str, file=f)
                    print(LGM_loss_str, file=f)
                    
                    if psnr > 50:
                        print("---------------------", file=f)
                    
               
                # also save the predicted splatters and the 

                # # add write images for splatter to optimize
                # pred_images = out['images_opt'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                # pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                # kiui.write_image(f'{opt.workspace}/eval_epoch_{epoch}/{i}_image_splatter_opt.jpg', pred_images)

                # pred_alphas = out['alphas_opt'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                # pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                # kiui.write_image(f'{opt.workspace}/eval_epoch_{epoch}/{i}_image_splatter_opt_alpha.jpg', pred_alphas)
                    
        
            torch.cuda.empty_cache()

        total_psnr /= num_samples_eval
        total_psnr_LGM /= num_samples_eval
        total_lpips /= num_samples_eval
        total_lpips_LGM /= num_samples_eval
        
        with open(f"{opt.workspace}/metrics.txt", "a") as f:
            # print(f"Total samples to eval = {num_samples_eval}", file=f)
            our_loss_str = f"Total - our_psnr = {total_psnr:.3f}, \t lpips = {total_lpips:.3f}"
            if opt.log_each_attribute_loss:
                for _attr in ordered_attr_list:
                    _loss_attr = total_attr_loss_dict[f'loss_{_attr}'] / num_samples_eval
                    our_loss_str += f" \t {_attr}: {_loss_attr:.3f}"
            print(our_loss_str, file=f)
            print(f"Total - LGM_psnr = {total_psnr_LGM:.3f}, \t lpips = {total_lpips_LGM:.3f}", file=f)
            
    

    # prof.export_chrome_trace("output_trace.json")
if __name__ == "__main__":
    
    # mp.set_start_method('spawn')
    ### Ignore the FutureWarning from pipeline_stable_diffusion.py
    warnings.filterwarnings("ignore", category=FutureWarning, module="pipeline_stable_diffusion")
    
    main()
    
    # Reset the warning filter to its default state (optional)
    warnings.resetwarnings()