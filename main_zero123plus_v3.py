import tyro
import time
import random

import torch
from core.options import AllConfigs
from core.models_zero123plus import Zero123PlusGaussian
from core.models_fix_pretrained import LGM

from accelerate import Accelerator, DistributedDataParallelKwargs
from safetensors.torch import load_file
from core.dataset_v3 import ObjaverseDataset as Dataset

import kiui
from datetime import datetime
import torch.utils.tensorboard as tensorboard
import shutil, os

from ipdb import set_trace as st

def main():    
    opt = tyro.cli(AllConfigs)

    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        # mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        # kwargs_handlers=[ddp_kwargs],
    )

    # model
    if opt.model_type == 'Zero123PlusGaussian':
        model = Zero123PlusGaussian(opt)
    elif opt.model_type == 'LGM':
        model = LGM(opt)
    # model = SingleSplatterImage(opt)
    # opt.workspace += datetime.now().strftime("%Y%m%d-%H%M%S")
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    loss_str = 'loss'
    assert (opt.lambda_rendering + opt.lambda_splatter + opt.lambda_lpips > 0), 'Must have at least one loss'
    if opt.lambda_rendering > 0:
        loss_str+=f'_render{opt.lambda_rendering}'
    if opt.lambda_splatter > 0:
        loss_str+=f'_splatter{opt.lambda_splatter}'
    if opt.lambda_lpips > 0:
        loss_str+=f'_lpips{opt.lambda_lpips}'
    desc = opt.desc
    if opt.train_unet:
        desc += '_train_unet'
    opt.workspace = os.path.join(opt.workspace, f"{time_str}-{desc}-{loss_str}-lr{opt.lr}")
    writer = tensorboard.SummaryWriter(opt.workspace)

    src_snapshot_folder = os.path.join(opt.workspace, 'src')
    ignore_func = lambda d, files: [f for f in files if f.endswith('__pycache__')]
    for folder in ['core', 'scripts', 'zero123plus']:
        dst_dir = os.path.join(src_snapshot_folder, folder)
        shutil.copytree(folder, dst_dir, ignore=ignore_func, dirs_exist_ok=True)
    for file in ['main_zero123plus_v3.py']:
        dest_file = os.path.join(src_snapshot_folder, file)
        shutil.copy2(file, dest_file)

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
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3000, eta_min=1e-6)
    total_steps = opt.num_epochs * len(train_dataloader)
    pct_start = 3000 / total_steps
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.lr, total_steps=total_steps, pct_start=pct_start)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3000, eta_min=1e-6)

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
        total_loss_lpips = 0 #torch.tensor([0])
        for i, data in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                optimizer.zero_grad()

                step_ratio = (epoch + i / len(train_dataloader)) / opt.num_epochs

                out = model(data, step_ratio)
                loss = out['loss']
                psnr = out['psnr']
                accelerator.backward(loss)

                # ## debug
                # # Check gradients of the unet parameters
                # print(f"check unet parameters")
                # for name, param in model.unet.named_parameters():
                #     if param.requires_grad and param.grad is not None:
                #         print(f"Parameter {name}, Gradient norm: {param.grad.norm().item()}")


                # gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)

                optimizer.step()
                scheduler.step()

                total_loss += loss.detach()
                total_psnr += psnr.detach()
                
                if 'loss_splatter' in out.keys():
                    total_loss_splatter += out['loss_splatter'].detach()
                if 'loss_rendering' in out.keys():
                    total_loss_rendering += out['loss_rendering'].detach()
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

        if 'loss_splatter' in out.keys():
            total_loss_splatter = accelerator.gather_for_metrics(total_loss_splatter).mean().item()
        if 'loss_rendering' in out.keys():
            total_loss_rendering = accelerator.gather_for_metrics(total_loss_rendering).mean().item()
        if 'loss_lpips' in out.keys():
            total_loss_lpips = accelerator.gather_for_metrics(total_loss_lpips).mean().item()
        if accelerator.is_main_process:
            total_loss /= len(train_dataloader)
            total_psnr /= len(train_dataloader)
            total_loss_splatter /= len(train_dataloader)
            total_loss_rendering /= len(train_dataloader)
            total_loss_lpips /= len(train_dataloader)
            
            accelerator.print(f"[train] epoch: {epoch} loss: {total_loss.item():.6f} psnr: {total_psnr.item():.4f} splatter_loss: {total_loss_splatter:.4f} rendering_loss: {total_loss_rendering:.4f} lpips_loss: {total_loss_lpips:.4f} ")
            writer.add_scalar('train/loss', total_loss.item(), epoch)
            writer.add_scalar('train/psnr', total_psnr.item(), epoch)
            writer.add_scalar('train/loss_splatter', total_loss_splatter, epoch)
            writer.add_scalar('train/loss_rendering', total_loss_rendering, epoch)
            writer.add_scalar('train/loss_lpips', total_loss_lpips, epoch)
        
        # checkpoint
        # if epoch % 10 == 0 or epoch == opt.num_epochs - 1:
        if epoch > 0 and epoch % opt.save_iter == 0:
            accelerator.wait_for_everyone()
            accelerator.save_model(model, opt.workspace)

        if epoch % opt.eval_iter == 0:
            # eval
            with torch.no_grad():
                model.eval()
                total_psnr = 0
                for i, data in enumerate(test_dataloader):

                    out = model(data)
        
                    psnr = out['psnr']
                    total_psnr += psnr.detach()
                    
                    # save some images
                    if accelerator.is_main_process:
                        gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                        kiui.write_image(f'{opt.workspace}/eval_gt_images_{epoch}_{i}.jpg', gt_images)

                        pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                        kiui.write_image(f'{opt.workspace}/eval_pred_images_{epoch}_{i}.jpg', pred_images)

                        # pred_alphas = out['alphas_pred'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                        # pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                        # kiui.write_image(f'{opt.workspace}/eval_pred_alphas_{epoch}_{i}.jpg', pred_alphas)
                
            
                torch.cuda.empty_cache()

                total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
                if accelerator.is_main_process:
                    total_psnr /= len(test_dataloader)
                    accelerator.print(f"[eval] epoch: {epoch} psnr: {psnr:.4f}")



if __name__ == "__main__":
    main()
