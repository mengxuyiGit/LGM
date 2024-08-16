import tyro
import time
import random

import torch
from core.options import AllConfigs
from core.models_lvis import LGM
from accelerate import Accelerator, DistributedDataParallelKwargs
from safetensors.torch import load_file

import kiui

from datetime import datetime
import re
import os
import torch.utils.tensorboard as tensorboard
from ipdb import set_trace as st
import numpy as np

from utils.general_utils import colormap

def save_dndn(render_pkg, path):
    depth = render_pkg["surf_depth"]
    norm = depth.max()
    depth = depth / norm

    depth = colormap(depth.detach().cpu().numpy()[0,:,0], cmap='turbo') # torch.Size([8, 3, 320, 320])
    depth = depth.detach().cpu().numpy()[None]
    
    surf_normal = render_pkg["surf_normal"].detach().cpu().numpy() * 0.5 + 0.5
    rend_normal = render_pkg["rend_normal"].detach().cpu().numpy() * 0.5 + 0.5

    # tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
    # tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
    # tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

    rend_dist = render_pkg["rend_dist"].detach().cpu().numpy()
    rend_dist = colormap(rend_dist[0,:,0])[None]
    # tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
    rend_dist = rend_dist.detach().cpu().numpy()
    

    # pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
    pred_images = np.concatenate([depth, surf_normal, rend_dist, rend_normal], axis=3)
    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[4], 3)
    kiui.write_image(path, pred_images)



def main():    
    opt = tyro.cli(AllConfigs)

    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        # kwargs_handlers=[ddp_kwargs],
    )
    
    # output folder
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
    
    if opt.desc is not None:
        time_str += f'_{opt.desc}'
        
    opt.workspace = os.path.join(opt.workspace, f'{time_str}')
    if accelerator.is_main_process:
        writer = tensorboard.SummaryWriter(opt.workspace)
    
    # model
    model = LGM(opt)

    # resume
    if opt.resume is not None:
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
    
    # data
    if opt.data_mode == 's3':
        from core.provider_lvis import ObjaverseDataset as Dataset
    else:
        raise NotImplementedError

    train_dataset = Dataset(opt, training=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
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
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3000, eta_min=1e-6)
    total_steps = opt.num_epochs * len(train_dataloader)
    pct_start = 3000 / total_steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.lr, total_steps=total_steps, pct_start=pct_start)

    # accelerate
    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, scheduler
    )

    # eval
    with torch.no_grad():
        model.eval()
        total_psnr = 0
        total_loss_mse = 0
        total_loss_lpips = 0
        for i, data in enumerate(test_dataloader):

            out = model(data)

            psnr = out['psnr']
            total_psnr += psnr.detach()
            total_loss_mse += out['loss_mse'].detach()
            total_loss_lpips += out['loss_lpips'].detach()
            
            # save some images
            if accelerator.is_main_process:
                epoch = "init"
                gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                kiui.write_image(f'{opt.workspace}/eval_gt_images_{epoch}_{i}.jpg', gt_images)

                pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                kiui.write_image(f'{opt.workspace}/eval_pred_images_{epoch}_{i}.jpg', pred_images)

                # pred_alphas = out['alphas_pred'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                # pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                # kiui.write_image(f'{opt.workspace}/eval_pred_alphas_{epoch}_{i}.jpg', pred_alphas)
                
                # save 2DGS depth and normal renderings
                if 'surf_normal' in out.keys():
                    save_dndn(out, path=f'{opt.workspace}/eval_pred_SdnRdns_{epoch}_{i}.jpg')

                   

        torch.cuda.empty_cache()

        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        if accelerator.is_main_process:
            total_psnr /= len(test_dataloader)
            total_loss_mse /= len(test_dataloader)
            total_loss_lpips /= len(test_dataloader)
            accelerator.print(f"[eval] epoch: {epoch} psnr: {psnr:.4f}")

            step = 0
            writer.add_scalar('eval/loss', (total_loss_mse + total_loss_lpips).item(), step)
            writer.add_scalar('eval/psnr', total_psnr.item(), step)
        

    # loop
    for epoch in range(opt.num_epochs):
        # train
        model.train()
        total_loss = 0
        total_psnr = 0

        log_loss = 0
        log_loss_mse = 0
        log_loss_lpips = 0
        log_psnr = 0
        log_loss_2dgs_dist = 0
        log_loss_2dgs_normal = 0
        
        for i, data in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                optimizer.zero_grad()

                step_ratio = (epoch + i / len(train_dataloader)) / opt.num_epochs

                
                # if (epoch * len(train_dataloader) + i) == 2000:
                #     accelerator.wait_for_everyone()
                #     os.makedirs(os.path.join(opt.workspace, "model_iteration_20000"), exist_ok=True)
                #     accelerator.save_model(model, os.path.join(opt.workspace, "model_iteration_2000"))
                #     accelerator.print(f"[INFO] saved model at iteration 2000")

                out = model(data, step_ratio, iteration=epoch * len(train_dataloader) + i)
                loss = out['loss']
                psnr = out['psnr']
                accelerator.backward(loss)

                # gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)

                optimizer.step()
                # scheduler.step()

                total_loss += loss.detach()
                total_psnr += psnr.detach()
                
                log_loss += loss.detach()
                log_psnr += psnr.detach()
                log_loss_mse += out['loss_mse'].detach()
                log_loss_lpips += out['loss_lpips'].detach()
                log_loss_2dgs_dist += out['dist_loss'].detach()
                log_loss_2dgs_normal += out['normal_loss'].detach()

            if accelerator.is_main_process:
                # logging
                log_iter = 10
                if i > 0 and i % log_iter ==0:
                    step = epoch * len(train_dataloader) + i
                    writer.add_scalar('train/loss', log_loss.item()/log_iter, step)
                    writer.add_scalar('train/psnr', log_psnr.item()/log_iter, step)
                    writer.add_scalar('train/loss_mse', log_loss_mse.item()/log_iter, step)
                    writer.add_scalar('train/loss_lpips', log_loss_lpips.item()/log_iter, step)        
                    writer.add_scalar('train/loss_2dgs_dist', log_loss_2dgs_dist.item()/log_iter, step)        
                    writer.add_scalar('train/loss_2dgs_normal', log_loss_2dgs_normal.item()/log_iter, step)
                    log_loss = 0
                    log_loss_mse = 0
                    log_loss_lpips = 0
                    log_psnr = 0
                    log_loss_2dgs_dist = 0
                    log_loss_2dgs_normal = 0
                    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], step)        
                    
                if i % 100 == 0:
                    mem_free, mem_total = torch.cuda.mem_get_info()    
                    print(f"[INFO] {i}/{len(train_dataloader)} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G lr: {scheduler.get_last_lr()[0]:.7f} step_ratio: {step_ratio:.4f} loss: {loss.item():.6f}")
                    
                # save log images
                if i % 500 == 0:
                    gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                    kiui.write_image(f'{opt.workspace}/train_gt_images_{epoch}_{i}.jpg', gt_images)

                    # gt_alphas = data['masks_output'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                    # gt_alphas = gt_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, gt_alphas.shape[1] * gt_alphas.shape[3], 1)
                    # kiui.write_image(f'{opt.workspace}/train_gt_alphas_{epoch}_{i}.jpg', gt_alphas)

                    pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                    kiui.write_image(f'{opt.workspace}/train_pred_images_{epoch}_{i}.jpg', pred_images)

                    # pred_alphas = out['alphas_pred'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                    # pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                    # kiui.write_image(f'{opt.workspace}/train_pred_alphas_{epoch}_{i}.jpg', pred_alphas)
                    
                    # save 2DGS depth and normal renderings
                    if 'surf_normal' in out.keys():
                        save_dndn(out, path=f'{opt.workspace}/train_pred_SdnRdn_{epoch}_{i}.jpg')


        # checkpoint
        if epoch % 2 == 0 or epoch == opt.num_epochs - 1:
        # if i % 100 == 0:
            accelerator.wait_for_everyone()
            # accelerator.save_model(model, opt.workspace)
            os.makedirs(os.path.join(opt.workspace, "model_epoch_{epoch}"), exist_ok=True)
            accelerator.save_model(model, os.path.join(opt.workspace, f"model_epoch_{epoch}"))
            

            # eval
            with torch.no_grad():
                model.eval()
                total_psnr_eval = 0
                total_loss_mse = 0
                total_loss_lpips = 0
                for j, data in enumerate(test_dataloader):

                    out = model(data)
        
                    psnr = out['psnr']
                    total_psnr_eval += psnr.detach()
                    total_loss_mse += out['loss_mse'].detach()
                    total_loss_lpips += out['loss_lpips'].detach()
                    
                    # save some images
                    if accelerator.is_main_process:
                        gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                        kiui.write_image(f'{opt.workspace}/eval_gt_images_{epoch}_{j}.jpg', gt_images)

                        pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                        kiui.write_image(f'{opt.workspace}/eval_pred_images_{epoch}_{j}.jpg', pred_images)

                        # pred_alphas = out['alphas_pred'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                        # pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                        # kiui.write_image(f'{opt.workspace}/eval_pred_alphas_{epoch}_{i}.jpg', pred_alphas)
                            
                        # save 2DGS depth and normal renderings
                        if 'surf_normal' in out.keys():
                            save_dndn(out, path=f'{opt.workspace}/eval_pred_SdnRdn_{epoch}_{j}.jpg')
                        

                torch.cuda.empty_cache()

                total_psnr_eval = accelerator.gather_for_metrics(total_psnr_eval).mean()
                if accelerator.is_main_process:
                    total_psnr_eval /= len(test_dataloader)
                    total_loss_mse /= len(test_dataloader)
                    total_loss_lpips /= len(test_dataloader)
                    accelerator.print(f"[eval] epoch: {epoch} psnr: {total_psnr_eval:.4f}")

                    step = (epoch + 1)* len(train_dataloader) 
                    writer.add_scalar('eval/loss', (total_loss_mse + total_loss_lpips).item(), step)
                    writer.add_scalar('eval/psnr', total_psnr_eval.item(), step)



        
        total_loss = accelerator.gather_for_metrics(total_loss).mean()
        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        if accelerator.is_main_process:
            total_loss /= len(train_dataloader)
            total_psnr /= len(train_dataloader)
            accelerator.print(f"[train] epoch: {epoch} loss: {total_loss.item():.6f} psnr: {total_psnr.item():.4f}")

   
if __name__ == "__main__":
    main()
