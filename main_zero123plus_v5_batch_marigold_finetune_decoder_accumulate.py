import tyro
import time
import random

import torch
from core.options import AllConfigs
from core.models_zero123plus_marigold_unet_rendering_loss_cross_domain import Zero123PlusGaussianMarigoldUnetCrossDomain, fuse_splatters
from core.dataset_v5_marigold import gt_attr_keys, start_indices, end_indices
from core.dataset_v5_marigold import ObjaverseDataset as Dataset
from core.dataset_lara import gobjverse as Dataset
# from core.provider_lara_splatter_optimized import gobjverse as Dataset

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
import multiprocessing as mp
import math
from accelerate.logging import get_logger
logger = get_logger(__name__, log_level="INFO")

from utils.format_helper import get_workspace_name, save_dndn
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
        folders = ['core', 'zero123plus']
        folders.append('sb_scripts') if opt.sbatch_launch else folders.append('scripts')
        for folder in folders:
            dst_dir = os.path.join(src_snapshot_folder, folder)
            shutil.copytree(folder, dst_dir, ignore=ignore_func, dirs_exist_ok=True)
        for file in ['main_zero123plus_v5_batch_marigold_finetune_decoder_accumulate.py']:
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
   
    torch.cuda.empty_cache()
    
    train_dataset = Dataset(opt, training=True)
    test_dataset = Dataset(opt, training=False)

    if opt.data_mode == 'lara':
        def random_init(id):
            torch.utils.data.get_worker_info().dataset.worker_init_open_db()

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            # num_workers=opt.num_workers,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=random_init,
            # persistent_workers=True,
        )
        
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            # num_workers=opt.num_workers,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=random_init,
            # persistent_workers=True,
        )
        
    else:
            
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        

    # optimizer
    print_grad_status(model, file_path=f"{opt.workspace}/model_grad_status_before.txt")
    print("before ")
    
    assert not (opt.finetune_decoder and opt.train_unet)
    if opt.finetune_decoder:
    
        parameters_list = []
        for name, para in model.vae.decoder.named_parameters():
            parameters_list.append(para)
            para.requires_grad = True
        
        if opt.decoder_with_domain_embedding:
            parameters_list.append(model.decoder_domain_embedding)
            model.decoder_domain_embedding.requires_grad = True
       
        
    elif opt.train_unet:
        # print_grad_status(model.unet, file_path=f"{opt.workspace}/model_grad_status_before.txt")
        # print("before ")
        
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
    
    else:
        raise NotImplementedError("Only train VAE or UNet")
                
    print_grad_status(model, file_path=f"{opt.workspace}/model_grad_status_after.txt")
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

    # actual batch size 
   
    total_batch_size = opt.batch_size * accelerator.num_processes * opt.gradient_accumulation_steps
    print('')
    logger.info(f"  Instantaneous batch size per device = {opt.batch_size }")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {opt.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {opt.max_train_steps}") # TODO: calculate total update steps
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / opt.gradient_accumulation_steps)
    opt.num_train_epochs = math.ceil(opt.max_train_steps / num_update_steps_per_epoch)
    initial_global_step = 0
    
    logger.info(f"  num_update_steps_per_epoch = {num_update_steps_per_epoch}")
    logger.info(f"  Total train num_train_epochs = {opt.num_train_epochs}")
   

    # loop
    # with torch.autograd.profiler.profile(use_cuda=True, record_shapes=True, profile_memory=True) as prof:
    if True:
        
        global_step = 0
        progress_bar = tqdm(
            range(0, opt.max_train_steps),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )
        
        
        for epoch in range(opt.num_train_epochs):
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
            total_loss_gt_normal = 0
            total_loss_normal = 0
            
            train_loss = 0.0
            
            if opt.log_gs_loss_mse_dict:
                # gt_attr_keys = ['pos', 'opacity', 'scale', 'rotation', 'rgbs']
                total_gs_loss_mse_dict = dict()
                for key in gt_attr_keys:
                    total_gs_loss_mse_dict[key] = 0
            
            # for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), disable=(opt.verbose_main), desc = f"Training epoch {epoch}"):
            for i, data in enumerate(train_dataloader): #, total=len(train_dataloader), disable=(opt.verbose_main), desc = f"Training epoch {epoch}"):
                if i > 0 and opt.skip_training:
                    break

                # if i > 5:
                #     st()
              
                if opt.verbose_main:
                    print(f"data['input']:{data['input'].shape}")
                    
                with accelerator.accumulate(model):
                    optimizer.zero_grad()
                    step_ratio = (epoch + i / len(train_dataloader)) / opt.num_train_epochs

                    # # Store initial weights before the update 
                    # initial_weights = store_initial_weights(model)

                    out = model(data, step_ratio, iteration=global_step)
                    # del data
                    loss = out['loss']
                    loss_splatter = out['loss_splatter'] if opt.finetune_decoder else torch.zeros_like(out['loss_latent'])
                    loss_latent = out['loss_latent'] if opt.train_unet else torch.zeros_like(loss)
                    loss_splatter_lpips = out['loss_splatter_lpips'] if 'loss_splatter_lpips' in out.keys() else torch.zeros_like(out['loss'])
                    # print("loss: ", loss, " loss_splatter: ", loss_splatter, "loss_latent: ", loss_latent, "loss_splatter_lpips", loss_splatter_lpips)
                    lossback = loss + loss_latent + loss_splatter + loss_splatter_lpips
                    # print("loss: ", loss, " loss_splatter: ", loss_splatter)
                    # lossback = loss + loss_latent + loss_splatter
                    accelerator.backward(lossback)
                    # print(f"epoch_{epoch}_iter_{i}: loss = {loss}")

                    # # debug
                    # if global_step > 0:
                    #     # Check gradients of the unet parameters
                    #     print(f"check unet parameters")
                    #     for name, param in model.unet.named_parameters():
                    #         if param.requires_grad and param.grad is not None:
                    #             print(f"Parameter {name}, Gradient norm: {param.grad.norm().item()}")
                    #     st()
                    
                    #     print(f"check other model parameters")
                    #     for name, param in model.named_parameters():
                    #         if param.requires_grad and param.grad is not None and "unet" not in name:
                    #             print(f"Parameter {name}, Gradient norm: {param.grad.norm().item()}")
                    #     st()
                    #     # TODO: CHECK decoder not have grad, especially deocder.others
                    #     # TODO: and check self.scale_bias

                    # gradient clipping
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)

                    optimizer.step()
                    if opt.lr_scheduler != 'Plat':
                        scheduler.step()
                    
                    # compare_weights(initial_weights=initial_weights, model=model)
                    # st()
                    
                    psnr = out['psnr']
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
                    if 'loss_gt_normal' in out.keys():
                        total_loss_gt_normal += out['loss_gt_normal'].detach()
                    if 'normal_loss' in out.keys():
                        total_loss_normal += out['normal_loss'].detach()


                # Log metrics after every step, not at the end of the epoch
                if accelerator.is_main_process:
                    writer.add_scalar('train/loss', loss.item(), global_step)
                    writer.add_scalar('train/psnr', psnr.item(), global_step)
                    writer.add_scalar('train/loss_latent', loss_latent.item(), global_step)
                    writer.add_scalar('train/loss_splatter', loss_splatter.item(), global_step)
                    writer.add_scalar('train/loss_lpips', out['loss_lpips'].item(), global_step) if 'loss_gt_normal' in out.keys() else None
                    writer.add_scalar('train/loss_gt_normal', out['loss_gt_normal'].detach().item(), global_step) if 'loss_gt_normal' in out.keys() else None
                    writer.add_scalar('train/loss_normal', out['normal_loss'].detach().item(), global_step) if 'normal_loss' in out.keys() else None
                
                # checkpoint
                # if epoch > 0 and epoch % opt.save_iter == 0:
                if global_step > 0 and global_step % opt.save_iter == 0 and not os.path.exists(os.path.join(opt.workspace, f"eval_global_step_{global_step}_ckpt")): # save by global step, not epoch
                    accelerator.wait_for_everyone()
                    accelerator.save_model(model, opt.workspace)
                    # save a copy 
                    accelerator.wait_for_everyone()
                    print("Saving a COPY of new ckpt ...")
                    accelerator.save_model(model, os.path.join(opt.workspace, f"eval_global_step_{global_step}_ckpt"))
                    print("Saved a COPY of new ckpt !!!")

                    torch.cuda.empty_cache()

                # if epoch % opt.eval_iter == 0: 
                # if global_step % opt.eval_iter == 0:  # eval by global step, not epoch
                if global_step % opt.eval_iter == 0 and not os.path.exists(os.path.join(opt.workspace, f"eval_global_step_{global_step}")):  # eval by global step, not epoch
                    # eval
                    accelerator.wait_for_everyone()
                    with torch.no_grad():
                        model.eval()
                
                        total_loss = 0
                        total_loss_latent = 0
                        total_psnr = 0
                        total_loss_splatter = 0 #torch.tensor([0]).to()
                        total_loss_rendering = 0 #torch.tensor([0])
                        total_loss_alpha = 0
                        total_loss_lpips = 0
                        total_loss_gt_normal = 0
                        total_loss_normal = 0
                        
                        print(f"Save to run dir: {opt.workspace}")
                        num_samples_eval = len(test_dataloader)
                        for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), disable=(opt.verbose_main), desc = f"Eval global_step {global_step}"):
                            if i > num_samples_eval:
                                break
                        
                            out = model(data, save_path=f'{opt.workspace}/eval_global_step_{global_step}', prefix=f"{accelerator.process_index}_{i}_", iteration=global_step)
                
                            psnr = out['psnr']
                            total_psnr += psnr.detach()
                            eval_loss = out['loss']
                            total_loss += eval_loss.detach()
                            loss_latent = out['loss_latent'] if opt.train_unet else torch.zeros_like(loss)
                            total_loss_latent += loss_latent.detach()
                            if 'loss_splatter' in out.keys():
                                total_loss_splatter += out['loss_splatter'].detach()
                            if 'loss_rendering' in out.keys():
                                total_loss_rendering += out['loss_rendering'].detach()
                            elif 'loss_alpha' in out.keys():
                                total_loss_alpha += out["loss_alpha"].detach()
                            if 'loss_lpips' in out.keys():
                                total_loss_lpips += out['loss_lpips'].detach()
                            if 'loss_gt_normal' in out.keys():
                                total_loss_gt_normal += out['loss_gt_normal'].detach()
                            if 'normal_loss' in out.keys():
                                total_loss_normal += out['normal_loss'].detach()
                            
                            # save some images
                            if True:
                                gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                                gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                                # kiui.write_image(f'{opt.workspace}/eval_epoch_{epoch}/{accelerator.process_index}_{i}_image_gt.jpg', gt_images)
                                kiui.write_image(f'{opt.workspace}/eval_global_step_{global_step}/{accelerator.process_index}_{i}_image_gt.jpg', gt_images)

                                pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                                pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                                # kiui.write_image(f'{opt.workspace}/eval_epoch_{epoch}/{accelerator.process_index}_{i}_image_pred.jpg', pred_images)
                                kiui.write_image(f'{opt.workspace}/eval_global_step_{global_step}/{accelerator.process_index}_{i}_image_pred.jpg', pred_images)

                                pred_alphas = out['alphas_pred'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                                pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                                # kiui.write_image(f'{opt.workspace}/eval_epoch_{epoch}/{accelerator.process_index}_{i}_image_alpha.jpg', pred_alphas)
                                kiui.write_image(f'{opt.workspace}/eval_global_step_{global_step}/{accelerator.process_index}_{i}_image_alpha.jpg', pred_alphas)
                                
                                # also save the predicted splatters and the 

                                # # add write images for splatter to optimize
                                # pred_images = out['images_opt'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                                # pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                                # kiui.write_image(f'{opt.workspace}/eval_epoch_{epoch}/{i}_image_splatter_opt.jpg', pred_images)

                                # pred_alphas = out['alphas_opt'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                                # pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                                # kiui.write_image(f'{opt.workspace}/eval_epoch_{epoch}/{i}_image_splatter_opt_alpha.jpg', pred_alphas)
                                
                                # save 2DGS depth and normal renderings
                                if 'surf_normal' in out.keys():
                                    save_dndn(out, path=f'{opt.workspace}/eval_global_step_{global_step}/{accelerator.process_index}_{i}_pred_SdnRdn.jpg')

                                
                                
                    
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
                            total_loss_gt_normal /= num_samples_eval
                            total_loss_normal /= num_samples_eval
                            
                            accelerator.print(f"[eval] epoch: {epoch} loss: {total_loss.item():.6f} loss_latent: {total_loss_latent.item():.6f} psnr: {total_psnr.item():.4f} splatter_loss: {total_loss_splatter:.4f} rendering_loss: {total_loss_rendering:.4f} alpha_loss: {total_loss_alpha:.4f} lpips_loss: {total_loss_lpips:.4f} ")
                            writer.add_scalar('eval/total_loss_latent', total_loss_latent.item(), global_step)
                            writer.add_scalar('eval/total_loss_other_than_latent', total_loss.item(), global_step)
                            writer.add_scalar('eval/total_psnr', total_psnr.item(), global_step)
                            writer.add_scalar('eval/total_loss_splatter', total_loss_splatter, global_step)
                            writer.add_scalar('eval/total_loss_rendering', total_loss_rendering, global_step)
                            writer.add_scalar('eval/total_loss_alpha', total_loss_alpha, global_step)
                            writer.add_scalar('eval/total_loss_lpips', total_loss_lpips, global_step)
                            writer.add_scalar('eval/total_loss_gt_normal', total_loss_gt_normal, global_step)
                            writer.add_scalar('eval/total_loss_normal', total_loss_normal, global_step)

                            if opt.lr_scheduler == 'Plat' and not opt.lr_schedule_by_train:
                                scheduler.step(total_loss)
                                writer.add_scalar('eval/lr', optimizer.param_groups[0]['lr'], epoch)
                    
                    # back to train mode to have grad
                    accelerator.wait_for_everyone()
                    model.train()
                
                if accelerator.sync_gradients:
                    # if args.use_ema:
                    #     ema_unet.step(unet.parameters())
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0
                
                # logs = {"step_loss": loss.detach().item(), "lr": optimizer.param_groups[0]['lr']} 
                if opt.finetune_decoder:
                    logs = {"step_loss": loss.detach().item(), "step_loss_splatter": loss_splatter.detach().item(), "lr": optimizer.param_groups[0]['lr']} 
                else:
                    logs = {"step_loss_latent": loss_latent.detach().item(), "lr": optimizer.param_groups[0]['lr']} 
                
                progress_bar.set_postfix(**logs)
                
                if global_step >= opt.max_train_steps:
                    break

    

    # prof.export_chrome_trace("output_trace.json")
if __name__ == "__main__":
    
    # mp.set_start_method('spawn')
    ### Ignore the FutureWarning from pipeline_stable_diffusion.py
    warnings.filterwarnings("ignore", category=FutureWarning, module="pipeline_stable_diffusion")
    
    main()
    
    # Reset the warning filter to its default state (optional)
    warnings.resetwarnings()
