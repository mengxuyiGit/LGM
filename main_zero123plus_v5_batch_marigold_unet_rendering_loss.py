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

import warnings
from accelerate.utils import broadcast
import re

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
  
    assert opt.model_type == "Zero123PlusGaussianMarigoldUnetCrossDomain", "Invalid model type"
    model =  Zero123PlusGaussianMarigoldUnetCrossDomain(opt)
    
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
        for file in ['main_zero123plus_v5_batch_marigold_unet_rendering_loss.py']:
            dest_file = os.path.join(src_snapshot_folder, file)
            shutil.copy2(file, dest_file)
        
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
    print_grad_status(model.unet, file_path=f"{opt.workspace}/model_grad_status_before.txt")
    print("before ")
    
    model.unet.requires_grad_(True)
    parameters_list = []
    if opt.only_train_attention:
        for name, para in model.unet.named_parameters():
            if 'transformer_blocks' in name:
                parameters_list.append(para)
                para.requires_grad = True
            else:
                para.requires_grad = False
    else:
        for name, para in model.unet.named_parameters():
            parameters_list.append(para)
            para.requires_grad = True
            
    print_grad_status(model.unet, file_path=f"{opt.workspace}/model_grad_status_after.txt")
    print("after ")
    
    optimizer = torch.optim.AdamW(parameters_list, lr=opt.lr, weight_decay=0.05, betas=(0.9, 0.95)) # TODO: lr can be 1e-3??
    

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
    

    # loop
    # with torch.autograd.profiler.profile(use_cuda=True, record_shapes=True, profile_memory=True) as prof:
    if True:
        for epoch in range(opt.num_epochs):
            torch.cuda.empty_cache()

            # train
            model.train()
            total_loss = 0
            total_loss_latent = 0
            total_psnr = 0
            total_loss_splatter = 0 #torch.tensor([0]).to()
            total_loss_rendering = 0 #torch.tensor([0])
            total_loss_alpha = 0
            total_loss_lpips = 0 #torch.tensor([0])
            
            if opt.log_gs_loss_mse_dict:
                # gt_attr_keys = ['pos', 'opacity', 'scale', 'rotation', 'rgbs']
                total_gs_loss_mse_dict = dict()
                for key in gt_attr_keys:
                    total_gs_loss_mse_dict[key] = 0
            
            splatter_guidance = (opt.lambda_splatter > 0) and (epoch <= opt.splatter_guidance_warmup) or (epoch % opt.splatter_guidance_interval == 0)
            # if splatter_guidance:
            #     print(f"splatter_guidance in epoch: {epoch}")
                    
            for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), disable=(opt.verbose_main), desc = f"Training epoch {epoch}"):
            
                if i > 0 and opt.skip_training:
                    break
              
                if opt.verbose_main:
                    print(f"data['input']:{data['input'].shape}")
                    
                with accelerator.accumulate(model):
                    optimizer.zero_grad()
                    step_ratio = (epoch + i / len(train_dataloader)) / opt.num_epochs

                    # # Store initial weights before the update
                    # initial_weights = store_initial_weights(model.unet)

                    out = model(data, step_ratio, splatter_guidance=splatter_guidance)
                    # st()
                    del data
                    loss = out['loss']
                    psnr = out['psnr']
                    loss_latent = out['loss_latent']

                    lossback = loss + loss_latent
                    accelerator.backward(lossback)
                    # print(f"epoch_{epoch}_iter_{i}: loss = {loss}")

                    # # debug
                    # # Check gradients of the unet parameters
                    # print(f"check unet parameters")
                    # for name, param in model.unet.named_parameters():
                    #     if param.requires_grad and param.grad is not None:
                    #         print(f"Parameter {name}, Gradient norm: {param.grad.norm().item()}")
                    # # st()
                
                    # print(f"check other model parameters")
                    # for name, param in model.named_parameters():
                    #     if param.requires_grad and param.grad is not None and "unet" not in name:
                    #         print(f"Parameter {name}, Gradient norm: {param.grad.norm().item()}")
                    # st()
                    # # TODO: CHECK decoder not have grad, especially deocder.others
                    # # TODO: and check self.scale_bias

                    # gradient clipping
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)

                    optimizer.step()
                    if opt.lr_scheduler != 'Plat':
                        scheduler.step()
                        
                    
                    total_loss += loss.detach()
                    total_psnr += psnr.detach()
                    total_loss_latent += loss_latent.detach()
                    
                    if 'loss_splatter' in out.keys():
                        total_loss_splatter += out['loss_splatter'].detach()
                    if 'loss_rendering' in out.keys():
                        total_loss_rendering += out['loss_rendering'].detach()
                    elif 'loss_alpha' in out.keys():
                        total_loss_alpha += out["loss_alpha"].detach()
                        
                    if 'loss_lpips' in out.keys():
                        total_loss_lpips += out['loss_lpips'].detach()


                # if accelerator.is_main_process:
                #     # logging
                #     if i % 100 == 0:
                #         mem_free, mem_total = torch.cuda.mem_get_info()    
                #         print(f"[INFO] {i}/{len(train_dataloader)} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G lr: {scheduler.get_last_lr()[0]:.7f} step_ratio: {step_ratio:.4f} loss: {loss.item():.6f}")
                    
                #     # save log images
                #     if i % 500 == 0:
                #         gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                #         gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                #         kiui.write_image(f'{opt.workspace}/train_gt_images_{epoch}_{i}.jpg', gt_images)

                #         # gt_alphas = data['masks_output'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                #         # gt_alphas = gt_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, gt_alphas.shape[1] * gt_alphas.shape[3], 1)
                #         # kiui.write_image(f'{opt.workspace}/train_gt_alphas_{epoch}_{i}.jpg', gt_alphas)

                #         pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                #         pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                #         kiui.write_image(f'{opt.workspace}/train_pred_images_{epoch}_{i}.jpg', pred_images)

                #         # pred_alphas = out['alphas_pred'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                #         # pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                #         # kiui.write_image(f'{opt.workspace}/train_pred_alphas_{epoch}_{i}.jpg', pred_alphas)
        
            total_loss = accelerator.gather_for_metrics(total_loss).mean()
            total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
            total_loss_latent = accelerator.gather_for_metrics(total_loss_latent).mean()
            if accelerator.is_main_process:
                total_loss /= len(train_dataloader)
                total_loss_latent /= len(train_dataloader)
                total_psnr /= len(train_dataloader)
                total_loss_splatter /= len(train_dataloader)
                total_loss_rendering /= len(train_dataloader)
                total_loss_alpha /= len(train_dataloader)
                total_loss_lpips /= len(train_dataloader)
                
                # TODO: only print this under verbose mode
                accelerator.print(f"[train] epoch: {epoch} loss: {total_loss.item():.6f} loss_latent: {total_loss_latent.item():.6f} psnr: {total_psnr.item():.4f} splatter_loss: {total_loss_splatter:.4f} rendering_loss: {total_loss_rendering:.4f} alpha_loss: {total_loss_alpha:.4f} lpips_loss: {total_loss_lpips:.4f} ")
                writer.add_scalar('train/loss_latent', total_loss_latent.item(), epoch) # for comparison with no rendering loss
                writer.add_scalar('train/loss_other_than_latent', total_loss.item(), epoch)
                writer.add_scalar('train/psnr', total_psnr.item(), epoch)
                writer.add_scalar('train/loss_splatter', total_loss_splatter, epoch)
                writer.add_scalar('train/loss_rendering', total_loss_rendering, epoch)
                writer.add_scalar('train/loss_alpha', total_loss_alpha, epoch)
                writer.add_scalar('train/loss_lpips', total_loss_lpips, epoch)
                if opt.lr_scheduler == 'Plat' and opt.lr_schedule_by_train:
                    scheduler.step(total_loss)
                    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
            
            
            # checkpoint
            # if epoch % 10 == 0 or epoch == opt.num_epochs - 1:
            if epoch > 0 and epoch % opt.save_iter == 0:
                accelerator.wait_for_everyone()
                accelerator.save_model(model, opt.workspace)
                
                # save a copy 
                accelerator.wait_for_everyone()
                print("Saving a COPY of new ckpt ...")
                accelerator.save_model(model, os.path.join(opt.workspace, f"eval_epoch_{epoch}"))
                print("Saved a COPY of new ckpt !!!")

                torch.cuda.empty_cache()

            if epoch % opt.eval_iter == 0: 
                # eval
                with torch.no_grad():
                    model.eval()
            
                    total_loss = 0
                    total_loss_latent = 0
                    total_psnr = 0
                    total_loss_splatter = 0 #torch.tensor([0]).to()
                    total_loss_rendering = 0 #torch.tensor([0])
                    total_loss_alpha = 0
                    total_loss_lpips = 0
                    
                    print(f"Save to run dir: {opt.workspace}")
                    num_samples_eval = 50
                    for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), disable=(opt.verbose_main), desc = f"Eval epoch {epoch}"):
                        if i > num_samples_eval:
                            break
                    
                        out = model(data, save_path=f'{opt.workspace}/eval_epoch_{epoch}', prefix=f"{accelerator.process_index}_{i}_")
                
                        psnr = out['psnr']
                        total_psnr += psnr.detach()
                        loss = out['loss']
                        total_loss += loss.detach()
                        loss_latent = out['loss_latent']
                        total_loss_latent += loss_latent.detach()
                        if 'loss_splatter' in out.keys():
                            total_loss_splatter += out['loss_splatter'].detach()
                        if 'loss_rendering' in out.keys():
                            total_loss_rendering += out['loss_rendering'].detach()
                        elif 'loss_alpha' in out.keys():
                            total_loss_alpha += out["loss_alpha"].detach()
                        if 'loss_lpips' in out.keys():
                            total_loss_lpips += out['loss_lpips'].detach()
                
                        
                        # save some images
                        if True:
                            gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                            gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                            kiui.write_image(f'{opt.workspace}/eval_epoch_{epoch}/{accelerator.process_index}_{i}_image_gt.jpg', gt_images)

                            pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                            pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                            kiui.write_image(f'{opt.workspace}/eval_epoch_{epoch}/{accelerator.process_index}_{i}_image_pred.jpg', pred_images)

                            pred_alphas = out['alphas_pred'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                            pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                            kiui.write_image(f'{opt.workspace}/eval_epoch_{epoch}/{accelerator.process_index}_{i}_image_alpha.jpg', pred_alphas)
                            
                            # also save the predicted splatters and the 

                            # # add write images for splatter to optimize
                            # pred_images = out['images_opt'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                            # pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                            # kiui.write_image(f'{opt.workspace}/eval_epoch_{epoch}/{i}_image_splatter_opt.jpg', pred_images)

                            # pred_alphas = out['alphas_opt'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                            # pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                            # kiui.write_image(f'{opt.workspace}/eval_epoch_{epoch}/{i}_image_splatter_opt_alpha.jpg', pred_alphas)
                            
                
                    torch.cuda.empty_cache()

                    total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
                    if accelerator.is_main_process:
                        total_psnr /= num_samples_eval
                        total_loss /= num_samples_eval
                        total_loss_latent /= num_samples_eval
                        total_loss_splatter /= num_samples_eval
                        total_loss_rendering /= num_samples_eval
                        total_loss_alpha /= num_samples_eval
                        total_loss_lpips /= num_samples_eval
                        
                        accelerator.print(f"[eval] epoch: {epoch} loss: {total_loss.item():.6f} loss_latent: {total_loss_latent.item():.6f} psnr: {total_psnr.item():.4f} splatter_loss: {total_loss_splatter:.4f} rendering_loss: {total_loss_rendering:.4f} alpha_loss: {total_loss_alpha:.4f} lpips_loss: {total_loss_lpips:.4f} ")
                        writer.add_scalar('eval/loss_latent', total_loss_latent.item(), epoch)
                        writer.add_scalar('eval/loss_other_than_latent', total_loss.item(), epoch)
                        writer.add_scalar('eval/psnr', total_psnr.item(), epoch)
                        writer.add_scalar('eval/loss_splatter', total_loss_splatter, epoch)
                        writer.add_scalar('eval/loss_rendering', total_loss_rendering, epoch)
                        writer.add_scalar('eval/loss_alpha', total_loss_alpha, epoch)
                        writer.add_scalar('eval/loss_lpips', total_loss_lpips, epoch)

                        if opt.lr_scheduler == 'Plat' and not opt.lr_schedule_by_train:
                            scheduler.step(total_loss)
                            writer.add_scalar('eval/lr', optimizer.param_groups[0]['lr'], epoch)
    

    # prof.export_chrome_trace("output_trace.json")
if __name__ == "__main__":
    
    ### Ignore the FutureWarning from pipeline_stable_diffusion.py
    warnings.filterwarnings("ignore", category=FutureWarning, module="pipeline_stable_diffusion")
    
    main()
    
    # Reset the warning filter to its default state (optional)
    warnings.resetwarnings()
