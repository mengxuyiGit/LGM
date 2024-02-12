import tyro
import time
import random

import torch
from core.options import AllConfigs
# from core.models import LGM
from core.models_fix_pretrained import LGM
from accelerate import Accelerator, DistributedDataParallelKwargs
from safetensors.torch import load_file

import kiui

from ipdb import set_trace as st
import os
import json
import re

def main():    
    opt = tyro.cli(AllConfigs)

    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        # kwargs_handlers=[ddp_kwargs],
    )

    # model
    model = LGM(opt)
    if opt.fix_pretrained:
        model.eval()
        # Freeze all parameters
        if opt.fix_pretrained:
            for name, param in model.named_parameters():
                if name=='splatter_out':
                    print(f"{name} still requires grad")
                else:
                    param.requires_grad = False

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
        from core.provider_objaverse import ObjaverseDataset as Dataset
    else:
        raise NotImplementedError

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

    # # optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.05, betas=(0.9, 0.95))
    # ---- new: optimizer ------
    if opt.fix_pretrained:
        # params_to_opt = filter(lambda p: p.requires_grad, model.parameters())
        # print(f"params_to_opt: {len(list(params_to_opt))}")
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, weight_decay=0.05, betas=(0.9, 0.95))
        print(f"opt.lr = {opt.lr}")
    
    else:
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

    # log with tb
    stats_jsonl = None
    stats_tfevents = None
    stats_metrics = dict()
    stats_dict = dict()
    rank = 0 # FIXME: get cuda device number when training using multiple GPUs
    cur_nimg = -1
    start_time = time.time()
    if rank == 0:
        ## ------------ dynamically change the workspace dir ------------
        prev_run_dirs = []
        outdir = opt.workspace
        if os.path.isdir(outdir):
            prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        desc = f"{opt.desc}-lr{opt.lr}"
        run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(run_dir)
    
        opt.workspace = run_dir
        # ------------ [end] ---------
        
        os.makedirs(opt.workspace, exist_ok=True)
        stats_jsonl = open(os.path.join(opt.workspace, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(opt.workspace)
        except ImportError as err:
            print('Skipping tfevents export:', err)


    # loop
    for epoch in range(opt.num_epochs):
        # train
        model.train()
        total_loss = 0
        total_psnr = 0
        for i, data in enumerate(train_dataloader):
            cur_nimg += 1
            with accelerator.accumulate(model):

                optimizer.zero_grad()

                step_ratio = (epoch + i / len(train_dataloader)) / opt.num_epochs

                out = model(data, step_ratio)
                loss = out['loss']
                psnr = out['psnr']
                
              
                # accelerator.backward(loss)
                # Backward pass
                if opt.fix_pretrained:
                    for param in model.parameters():
                        if param.requires_grad:
                            accelerator.backward(loss)
                else:
                    accelerator.backward(loss)

                # gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip) 

                optimizer.step()
                scheduler.step()
                
                # Print the parameter list and their gradients
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f"Parameter: {name}, Gradient: {param.grad.max(), param.grad.mean()}")
                    else:
                        # print(f"Parameter: {name}, Gradient: None")
                        pass

                total_loss += loss.detach()
                total_psnr += psnr.detach()
              
                
                ## log with tb
                timestamp = time.time()
                stats_metrics.update({
                    'loss':total_loss,
                    'psnr':total_psnr
                })
                stats_dict.update({
                    'loss':total_loss.item(),
                    'psnr':total_psnr.item()
                })
                
                if stats_jsonl is not None:
                    fields = dict(stats_dict, timestamp=timestamp)
                    stats_jsonl.write(json.dumps(fields) + '\n')
                    stats_jsonl.flush()
                if stats_tfevents is not None:
                    global_step = int(cur_nimg / 10)
                    walltime = timestamp - start_time
                    for name, value in stats_metrics.items():
                        stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
                    stats_tfevents.flush()
                    print("tf log sucessful!")

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
        if accelerator.is_main_process:
            total_loss /= len(train_dataloader)
            total_psnr /= len(train_dataloader)
            accelerator.print(f"[train] epoch: {epoch} loss: {total_loss.item():.6f} psnr: {total_psnr.item():.4f}")
        
        # checkpoint
        # if epoch % 10 == 0 or epoch == opt.num_epochs - 1:
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
