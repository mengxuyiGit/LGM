import tyro
import time
import random

import torch
from core.options import AllConfigs
from core.models_zero123plus import Zero123PlusGaussian, gt_attr_keys, start_indices, end_indices, fuse_splatters
from core.models_zero123plus_code import Zero123PlusGaussianCode

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

    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        # kwargs_handlers=[ddp_kwargs],
    )

    # model
    if opt.model_type == 'Zero123PlusGaussian':
        model = Zero123PlusGaussian(opt)
        from core.dataset_v4_batch import ObjaverseDataset as Dataset
    elif opt.model_type == 'Zero123PlusGaussianCode':
        model = Zero123PlusGaussianCode(opt)
        from core.dataset_v4_code import ObjaverseDataset as Dataset
    
    elif opt.model_type == 'LGM':
        model = LGM(opt)
    # model = SingleSplatterImage(opt)
    # opt.workspace += datetime.now().strftime("%Y%m%d-%H%M%S")
    # if accelerator.is_main_process:
    #     time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    #     print(f"main process time string: {time_str}")
    #     time_tensor = torch.tensor([ord(c) for c in time_str], dtype=torch.int64)
    
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
    if opt.splatter_guidance_interval > 0:
        desc += f"-sp_guide_{opt.splatter_guidance_interval}"
    if opt.codes_from_encoder:
        desc += "-codes_from_encoder"
    else:
        optimizer_cfg = opt.optimizer.copy()
        desc += f"-codes_lr{optimizer_cfg['lr']}"
        
    desc += f"-{opt.decoder_mode}"
    if opt.decode_splatter_to_128:
        desc += "-pred128"
        if opt.decoder_upblocks_interpolate_mode is not None:
            desc += f"_{opt.decoder_upblocks_interpolate_mode}"
            if opt.decoder_upblocks_interpolate_mode!="last_layer" and opt.replace_interpolate_with_avgpool:
                desc += "_avgpool"
        
    ## the following may not exists, thus directly added to opt.desc if exists
    if len(opt.attr_use_logrithm_loss) > 0:
        loss_special = '-logrithm'
        for key in opt.attr_use_logrithm_loss:
            loss_special += f"_{key}"
        desc += loss_special
    
    if len(opt.normalize_scale_using_gt) > 0:
        loss_special = '-norm'
        for key in opt.normalize_scale_using_gt:
            loss_special += f"_{key}"
        desc += loss_special
        
    if opt.train_unet:
        desc += '-train_unet'
    if opt.skip_predict_x0:
        desc += '-skip_predict_x0'
    if opt.num_views != 20:
        desc += f'-numV{opt.num_views}'
    
    opt.workspace = os.path.join(opt.workspace, f"{time_str}-{desc}-{loss_str}-lr{opt.lr}-{opt.lr_scheduler}")
    if opt.lr_scheduler == 'Plat':
            opt.workspace += f"{opt.lr_scheduler_patience}"
    
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
    
    if not opt.codes_from_encoder:
        opt.code_dir = os.path.join(opt.workspace, 'code_dir')
        print(f"Codes are saved to:{opt.code_dir}")

    if accelerator.is_main_process:
        src_snapshot_folder = os.path.join(opt.workspace, 'src')
        ignore_func = lambda d, files: [f for f in files if f.endswith('__pycache__')]
        for folder in ['core', 'scripts', 'zero123plus']:
            dst_dir = os.path.join(src_snapshot_folder, folder)
            shutil.copytree(folder, dst_dir, ignore=ignore_func, dirs_exist_ok=True)
        for file in ['main_zero123plus_v4_batch_code.py']:
            dest_file = os.path.join(src_snapshot_folder, file)
            shutil.copy2(file, dest_file)
        
    # resume
    if opt.resume is not None and not opt.decoder_mode.startswith("v1_fix_rgb"):
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
                    state_dict[k].copy_(v)
                else:
                    accelerator.print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
            else:
                accelerator.print(f'[WARN] unexpected param {k}: {v.shape}')
    
    train_dataset = Dataset(opt, training=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
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
    
   
    # loop
    for epoch in range(opt.num_epochs):
        # train
        model.train()
        total_loss = 0
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
        if splatter_guidance:
            print(f"splatter_guidance in epoch: {epoch}")
                
        # for i, data in enumerate(train_dataloader):
        for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), disable=(opt.verbose_main), desc = f"Training epoch {epoch}"):
            if i > 0 and opt.skip_training:
                break
                
            if opt.verbose_main:
                print(f"data['input']:{data['input'].shape}")
                
            with accelerator.accumulate(model):
                if not opt.codes_from_encoder:
                    ## ---- load or init code here ----
                    if num_gpus==1:
                        codes_before_act_list_grad_, codes, code_optimizers = model.load_scenes(opt.code_dir, data)
                    else:
                        codes_before_act_list_grad_, codes, code_optimizers = model.module.load_scenes(opt.code_dir, data)
                    for code_optimizer in code_optimizers:
                        # print("code_optimizer.zero_grad()")
                        code_optimizer.zero_grad()
                    
                    data['codes'] = codes
               
                    # ---- finish code init ----

                optimizer.zero_grad()

                step_ratio = (epoch + i / len(train_dataloader)) / opt.num_epochs

                
                out = model(data, step_ratio, splatter_guidance=splatter_guidance, epoch=epoch)
                loss = out['loss']
                psnr = out['psnr']
               
                accelerator.backward(loss)
                # print(f"epoch_{epoch}_iter_{i}: loss = {loss}")

                # # debug
                # # Check gradients of the unet parameters
                # print(f"check unet parameters")
                # for name, param in model.unet.named_parameters():
                #     if param.requires_grad and param.grad is not None:
                #         print(f"Parameter {name}, Gradient norm: {param.grad.norm().item()}")
              
                # print(f"check other model parameters")
                # for name, param in model.named_parameters():
                #     if param.requires_grad and param.grad is not None:
                #         print(f"Parameter {name}, Gradient norm: {param.grad.norm().item()}")
                # st()

                # gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)

                optimizer.step()
                if opt.lr_scheduler != 'Plat':
                    scheduler.step()
                
                ## optimize and save code here
                if not opt.codes_from_encoder:
                    ## 1. do optimization step 
                    # --- for codes ---
                    if epoch > 0:
                        for code_optimizer in code_optimizers: # NOTE: value changed
                            code_optimizer.step()
                    else:  
                        before_optimization_params = [codes_before_act.clone().detach() for codes_before_act in codes_before_act_list_grad_]
                        
                        ### insert optimization step
                        for code_optimizer in code_optimizers: # NOTE: value changed
                            code_optimizer.step()
                        
                        after_optimization_params = [codes_before_act.clone().detach() for codes_before_act in codes_before_act_list_grad_]
                        parameters_changed = any(
                            not torch.equal(before, after) for before, after in zip(before_optimization_params, after_optimization_params)
                        )
                        if parameters_changed:
                            print("Parameters have changed after optimization step.")
                        else:
                            print("Parameters have not changed after optimization step. Are you sure you do not optimize code?")
                            st()
                
                    ## 2. save optimized code and splatter images
                    if num_gpus==1:
                        model.save_scenes(opt.code_dir, code_list_=codes_before_act_list_grad_, 
                                    scene_names=data['scene_name'],
                                    code_optimizer_list=code_optimizers
                                    )
                    
                    else:
                        model.module.save_scenes(opt.code_dir, code_list_=codes_before_act_list_grad_, 
                                    scene_names=data['scene_name'],
                                    code_optimizer_list=code_optimizers)
                    
                    # --- finish saving the code ----
                
                total_loss += loss.detach()
                total_psnr += psnr.detach()
                
                if 'loss_splatter' in out.keys():
                    total_loss_splatter += out['loss_splatter'].detach()
                if 'loss_rendering' in out.keys():
                    total_loss_rendering += out['loss_rendering'].detach()
                elif 'loss_alpha' in out.keys():
                    total_loss_alpha += out["loss_alpha"].detach()
                    
                if 'loss_lpips' in out.keys():
                    total_loss_lpips += out['loss_lpips'].detach()
              
                if opt.log_gs_loss_mse_dict:
                    for key in gt_attr_keys:
                        total_gs_loss_mse_dict[key] += out['gs_loss_mse_dict'][key].detach()
    

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

        if 'loss_splatter' in out.keys():
            total_loss_splatter = accelerator.gather_for_metrics(total_loss_splatter).mean().item()
        if 'loss_rendering' in out.keys():
            total_loss_rendering = accelerator.gather_for_metrics(total_loss_rendering).mean().item()
        elif 'loss_alpha' in out.keys():
            total_loss_alpha = accelerator.gather_for_metrics(total_loss_alpha).mean().item()
        
        if 'loss_lpips' in out.keys():
            total_loss_lpips = accelerator.gather_for_metrics(total_loss_lpips).mean().item()
        if opt.log_gs_loss_mse_dict:
            for key in gt_attr_keys:
                total_gs_loss_mse_dict[key] = accelerator.gather_for_metrics(total_gs_loss_mse_dict[key]).mean().item()
                
        if accelerator.is_main_process:
            total_loss /= len(train_dataloader)
            total_psnr /= len(train_dataloader)
            total_loss_splatter /= len(train_dataloader)
            total_loss_rendering /= len(train_dataloader)
            total_loss_alpha /= len(train_dataloader)
            total_loss_lpips /= len(train_dataloader)
            
            accelerator.print(f"[train] epoch: {epoch} loss: {total_loss.item():.6f} psnr: {total_psnr.item():.4f} splatter_loss: {total_loss_splatter:.4f} rendering_loss: {total_loss_rendering:.4f} alpha_loss: {total_loss_alpha:.4f} lpips_loss: {total_loss_lpips:.4f} ")
            writer.add_scalar('train/loss', total_loss.item(), epoch)
            writer.add_scalar('train/psnr', total_psnr.item(), epoch)
            writer.add_scalar('train/loss_splatter', total_loss_splatter, epoch)
            writer.add_scalar('train/loss_rendering', total_loss_rendering, epoch)
            writer.add_scalar('train/loss_alpha', total_loss_alpha, epoch)
            writer.add_scalar('train/loss_lpips', total_loss_lpips, epoch)
            if opt.log_gs_loss_mse_dict:
                for key in gt_attr_keys:
                    total_attr_loss = total_gs_loss_mse_dict[key]
                    # if key in opt.attr_use_logrithm_loss:
                    #     total_attr_loss = total_gs_loss_mse_dict[f"{key}_before_log"]
                    #     print(f"we log {key}_before_log")
                    # else:
                    #     total_attr_loss = total_gs_loss_mse_dict[key]
                    writer.add_scalar(f'train/loss(weighted)_{key}', total_attr_loss, epoch)
            
            if opt.lr_scheduler == 'Plat' and opt.lr_schedule_by_train:
                scheduler.step(total_loss)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
            
        
        # checkpoint
        # if epoch % 10 == 0 or epoch == opt.num_epochs - 1:
        if epoch > 0 and epoch % opt.save_iter == 0:
            # update new ckpt
            accelerator.wait_for_everyone()
            print("Saving new ckpt ...")
            accelerator.save_model(model, opt.workspace)
            print("Saved new ckpt !!!")
            
            if opt.save_ckpt_copies:
                # save a copy 
                accelerator.wait_for_everyone()
                print("Saving a COPY of new ckpt ...")
                accelerator.save_model(model, os.path.join(opt.workspace, f"eval_epoch_{epoch}"))
                print("Saved a COPY of new ckpt !!!")

        if epoch % opt.eval_iter == 0: 
            # eval
            with torch.no_grad():
                model.eval()
        
                total_loss = 0
                total_psnr = 0
                total_loss_splatter = 0 #torch.tensor([0]).to()
                total_loss_rendering = 0 #torch.tensor([0])
                total_loss_alpha = 0
                total_loss_lpips = 0
                
                print(f"Save to run dir: {opt.workspace}")
                for i, data in enumerate(test_dataloader):
                    if not opt.codes_from_encoder:
                        ## ---- load or init code here ----
                        if num_gpus==1:
                            codes = model.load_scenes(opt.code_dir, data, eval_mode=True)
                        else:
                            codes = model.module.load_scenes(opt.code_dir, data, eval_mode=True)
                        
                        data['codes'] = codes
                        
                        # ---- finish code init ----

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
                        kiui.write_image(f'{opt.workspace}/eval_epoch_{epoch}/{accelerator.process_index}_{i}_image_gt.jpg', gt_images)

                        pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                        kiui.write_image(f'{opt.workspace}/eval_epoch_{epoch}/{accelerator.process_index}_{i}_image_pred.jpg', pred_images)

                        pred_alphas = out['alphas_pred'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                        pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                        kiui.write_image(f'{opt.workspace}/eval_epoch_{epoch}/{accelerator.process_index}_{i}_image_alpha.jpg', pred_alphas)

                        # # add write images for splatter to optimize
                        # pred_images = out['images_opt'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        # pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                        # kiui.write_image(f'{opt.workspace}/eval_epoch_{epoch}/{i}_image_splatter_opt.jpg', pred_images)

                        # pred_alphas = out['alphas_opt'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                        # pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                        # kiui.write_image(f'{opt.workspace}/eval_epoch_{epoch}/{i}_image_splatter_opt_alpha.jpg', pred_alphas)

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
                                    name = f'histogram_batch{accelerator.process_index}_{i}_{splatters_pred_key}_{attr}'
                                    if attr == "scale":
                                        name += f"_{opt.scale_act}_bias{opt.scale_act_bias}"
                                    
                                    if opt.normalize_scale_using_gt:
                                        name += "normed_on_gt"
                                        
                                    plt.title(f'{name}')
                                    plt.savefig(f'{opt.workspace}/eval_epoch_{epoch}/{name}.jpg')
                                
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
                    
                    accelerator.print(f"[eval] epoch: {epoch} loss: {total_loss.item():.6f} psnr: {total_psnr.item():.4f} splatter_loss: {total_loss_splatter:.4f} rendering_loss: {total_loss_rendering:.4f} alpha_loss: {total_loss_alpha:.4f} lpips_loss: {total_loss_lpips:.4f} ")
                    writer.add_scalar('eval/loss', total_loss.item(), epoch)
                    writer.add_scalar('eval/psnr', total_psnr.item(), epoch)
                    writer.add_scalar('eval/loss_splatter', total_loss_splatter, epoch)
                    writer.add_scalar('eval/loss_rendering', total_loss_rendering, epoch)
                    writer.add_scalar('eval/loss_alpha', total_loss_alpha, epoch)
                    writer.add_scalar('eval/loss_lpips', total_loss_lpips, epoch)

                    if opt.lr_scheduler == 'Plat' and not opt.lr_schedule_by_train:
                        scheduler.step(total_loss)
                        writer.add_scalar('eval/lr', optimizer.param_groups[0]['lr'], epoch)
               
                if opt.save_train_pred > 0:
                    for j, data in enumerate(train_dataloader):
                        if j > opt.save_train_pred:
                            break
                        
                        if not opt.codes_from_encoder:
                            ## ---- load or init code here ----
                            if num_gpus==1:
                                codes = model.load_scenes(opt.code_dir, data, eval_mode=True)
                            else:
                                codes = model.module.load_scenes(opt.code_dir, data, eval_mode=True)
                            
                            data['codes'] = codes
                            
                            # ---- finish code init ----

                        out = model(data)
        
                        # save some training images
                        if accelerator.is_main_process:
                            gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                            gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                            kiui.write_image(f'{opt.workspace}/eval_epoch_{epoch}/{i+j}_train_image_gt.jpg', gt_images)

                            pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                            pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                            kiui.write_image(f'{opt.workspace}/eval_epoch_{epoch}/{i+j}_train_image_pred.jpg', pred_images)

                            pred_alphas = out['alphas_pred'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                            pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                            kiui.write_image(f'{opt.workspace}/eval_epoch_{epoch}/{i+j}_train_image_alpha.jpg', pred_alphas)

                            # # add write images for splatter to optimize
                            # pred_images = out['images_opt'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                            # pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                            # kiui.write_image(f'{opt.workspace}/eval_epoch_{epoch}/{i+j}_train_image_splatter_opt.jpg', pred_images)

                            # pred_alphas = out['alphas_opt'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                            # pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                            # kiui.write_image(f'{opt.workspace}/eval_epoch_{epoch}/{i+j}_train_image_splatter_opt_alpha.jpg', pred_alphas)



if __name__ == "__main__":
    
    ### Ignore the FutureWarning from pipeline_stable_diffusion.py
    warnings.filterwarnings("ignore", category=FutureWarning, module="pipeline_stable_diffusion")
    
    main()
    
    # Reset the warning filter to its default state (optional)
    warnings.resetwarnings()