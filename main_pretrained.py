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

    # data_name = '/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/0c58250e3a7242e9bf21b114f2c8dce6' # elephant
    if opt.data_path is not None:
        data_name = opt.data_path
    else:
        data_name = '/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/0c58250e3a7242e9bf21b114f2c8dce6' # elephant
    train_dataset = Dataset(opt, name=data_name, training=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        # num_workers=opt.num_workers,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    test_dataset = Dataset(opt, name=data_name, training=False)
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
    ## IMPORTANT: run a warm up iteration to update the parameters to the target value
    # device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)
    # model.eval()
    # accelerate
    fake_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    fake_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(fake_optimizer, T_0=3000, eta_min=1e-6)
    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, fake_optimizer, train_dataloader, test_dataloader, fake_scheduler
    )
    for i, data in enumerate(train_dataloader):
        out = model(data)
    # # Now you know the initial value of the dynamic parameter
    # initial_value = model.splatter_out
    # print(f"Initial value of dynamic parameter: {initial_value}")
    # initial_value = model.splatter_out.shape
    # print(f"Initial value of dynamic parameter (shape): {initial_value}")
    # # st()
    ## optimizer
    if opt.fix_pretrained:
        # params_to_opt = filter(lambda p: p.requires_grad, model.parameters())
        # print(f"params_to_opt: {len(list(params_to_opt))}")
        if opt.use_adamW:
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, weight_decay=0.05, betas=(0.9, 0.95))
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
            print(f"opt.lr = {opt.lr}, use normal Adam")
    
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.05, betas=(0.9, 0.95))
    
    if opt.lr_scheduler == 'CosAnn':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3000, eta_min=1e-6)
    elif opt.lr_scheduler == 'OneCyc':
        total_steps = opt.num_epochs * len(train_dataloader)
        pct_start = 3000 / total_steps
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.lr, total_steps=total_steps, pct_start=pct_start)
    elif opt.lr_scheduler == 'Plat':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=opt.lr_scheduler_factor, patience=opt.lr_scheduler_patience, verbose=True, min_lr=1e-6)
    else:
        assert ValueError('Not a valid lr_scheduler option.')

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
        desc = f"splat{opt.splat_size}-inV{opt.num_input_views}-lossV{opt.num_views}-lr{opt.lr}"
        if opt.use_adamW:
            desc = f"adamW-{desc}"
            
        
        if opt.lr_scheduler == 'Plat':
            desc = f"{opt.lr_scheduler}-patience_{opt.lr_scheduler_patience}-factor_{opt.lr_scheduler_factor}-eval_{opt.eval_iter}-{desc}"
        else:
            desc = f"{opt.lr_scheduler}-{desc}"
            
        if opt.desc is not None:
            desc = f"{opt.desc}-{desc}"
        run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
        print(f"[Save dir] {run_dir}")
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

                
                # print(f"-------1. before out=model():---------")
                # last_time = time.time()
                
                out = model(data, step_ratio, opt=opt, epoch=epoch, i=i)
                loss = out['loss']
                psnr = out['psnr']
                

                # print(f"-------2. before backward:{time.time()-last_time}---------")
                # last_time = time.time()
                
                
                # # accelerator.backward(loss)
                # # Backward pass
                # if opt.fix_pretrained:
                #     for param in model.parameters():
                #         if param.requires_grad:
                #             accelerator.backward(loss)
                # else:
                #     accelerator.backward(loss)
                # before_back_grad = None
                # if model.splatter_out.grad is not None:
                #     before_back_grad = model.splatter_out.grad.max()
                    
                # print(f"Parameter: splatter out, Gradient: {before_back_grad}")
                accelerator.backward(loss)
                # print("accelerator.backward(loss)")
                # # Print information about parameter groups
                # for i, param_group in enumerate(optimizer.param_groups):
                #     print(f"Parameter Group {i + 1}:")
                #     print(f"  Learning Rate: {param_group['lr']}")
                #     print(f"  Weight Decay: {param_group['weight_decay']}")
                #     print(f"  Parameters:")
                #     for param in param_group['params']:
                #         print(f"    {param.shape}")
                #     print()
                # st()

                # gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip) 

                # old = model.splatter_out.detach().clone().cpu()
                # print(f"splatter out before opt step:{old}")
                # st()
                optimizer.step()
               
                # scheduler.step()# FIXME: THIS is the original position of scheduler
                
                # new = old + opt.lr * model.splatter_out.grad
                
                # new = model.splatter_out.detach().clone().cpu()
                # # print(f"splatter out after opt step:{new}")
                # print(f"splatter out unchanges (use opt)?:{torch.all(old == new)}")
                # # st()
                
                # # Print the parameter list and their gradients
                # for name, param in model.named_parameters():
                #     if param.grad is not None:
                #         print(f"Parameter: {name}, Gradient: {param.grad.max(), param.grad.mean()}")
                #     else:
                #         # print(f"Parameter: {name}, Gradient: None")
                #         pass
                
                # print(f"-------3. before tb log:{time.time()-last_time}---------")
                # last_time = time.time()

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
                        stats_tfevents.add_scalar(f'Train/{name}', value, global_step=global_step, walltime=walltime)
                    stats_tfevents.flush()
                    # print("tf log sucessful!")
                
                print(f"stats_metrics:{stats_metrics}, stats_dict:{stats_dict}")
       
                

            # if accelerator.is_main_process:
            #     # logging
            #     # if i % 100 == 0:
            #     if i % 1 == 0:
            #         mem_free, mem_total = torch.cuda.mem_get_info()    
            #         print(f"[INFO] {i}/{len(train_dataloader)} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G lr: {scheduler.get_last_lr()[0]:.7f} step_ratio: {step_ratio:.4f} loss: {loss.item():.6f}")
                
            #     # save log images
            #     # if i % 500 == 0:
            #     if i % 1 == 0:
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

        # print(f"-------4. epoch end(before gather):{time.time()-last_time}---------")
        # last_time = time.time()
        
        total_loss = accelerator.gather_for_metrics(total_loss).mean()
        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        if accelerator.is_main_process:
            total_loss /= len(train_dataloader)
            total_psnr /= len(train_dataloader)
            accelerator.print(f"[train] epoch: {epoch} loss: {total_loss.item():.6f} psnr: {total_psnr.item():.4f}")
        
        # checkpoint
        # if epoch % 10 == 0 or epoch == opt.num_epochs - 1:
        accelerator.wait_for_everyone()
        if not opt.fix_pretrained:
            accelerator.save_model(model, opt.workspace)

        # print(f"-------5. epoch end(after save_model):{time.time()-last_time}---------")
        # last_time = time.time()

        if epoch % opt.eval_iter == 0:
            # eval
            with torch.no_grad():
                
                # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                # model = model.half().to(device) # TODO:
                
                model.eval()
                total_psnr = 0
                total_loss = 0
                for i, data in enumerate(test_dataloader):
                    # st()
                    # print(f"test data vids:{[t.item() for t in data['vids']]}")
                    out = model(data, opt=opt)
                   
                    psnr = out['psnr']
                    total_psnr += psnr.detach()
                    loss = out['loss']
                    total_loss += loss.detach()
                    
                    
                    # save some images
                    if accelerator.is_main_process and epoch % opt.save_iter == 0:
                        gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                        kiui.write_image(f'{opt.workspace}/eval_gt_images_{epoch}_{i}.jpg', gt_images)

                        pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                        kiui.write_image(f'{opt.workspace}/eval_pred_images_{epoch}_{i}.jpg', pred_images)
                        # st()

                        # pred_alphas = out['alphas_pred'].detach().cpu().numpy() # [B, V, 1, output_size, output_size]
                        # pred_alphas = pred_alphas.transpose(0, 3, 1, 4, 2).reshape(-1, pred_alphas.shape[1] * pred_alphas.shape[3], 1)
                        # kiui.write_image(f'{opt.workspace}/eval_pred_alphas_{epoch}_{i}.jpg', pred_alphas)
                        
                        # save optimized ply
                        if not os.path.exists(os.path.join(opt.workspace, f'eval_pred_gs_{epoch}_{i}')):
                            os.makedirs(os.path.join(opt.workspace, f'eval_pred_gs_{epoch}_{i}'))
                        
                        ## save spaltter imgs: model.splatter_out
                        splatter_out_save_batch = model.get_activated_splatter_out()
                        for splatter_out_save in splatter_out_save_batch:
                            for j, _sp_im in enumerate(splatter_out_save):
                                model.gs.save_ply(_sp_im[None], os.path.join(opt.workspace, f'eval_pred_gs_{epoch}_{i}', f'splatter_{j}' + '.ply')) # print(_sp_im[None].shape) # [1, splatter_res**2, 14]
                                # # load_path = os.path.join(opt.workspace, f'eval_pred_gs_{epoch}_{i}', f'splatter_{j}' + '.ply')
                                # load_path = '/home/xuyimeng/Repo/LGM/runs/LGM_optimize_splatter/workspace_splatter_gt_full_ply/00000-hydrant-inV6-lossV20-lr0.0006/eval_pred_gs_100_0/splatter_2.ply'
                                # gaussians_loaded = model.gs.load_ply(load_path)
                                # st() # then peroform einops, and check
                        ## save fused gaussian
                        model.gs.save_ply(out['gaussians'].detach(), os.path.join(opt.workspace, f'eval_pred_gs_{epoch}_{i}', 'fused' + '.ply')) # out['gaussians'].shape: [B, Npts, 14]


                torch.cuda.empty_cache()

                total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
                total_loss = accelerator.gather_for_metrics(total_loss).mean()
                if accelerator.is_main_process:
                    total_psnr /= len(test_dataloader)
                    total_loss /= len(test_dataloader)
                    accelerator.print(f"[eval] epoch: {epoch} psnr: {psnr:.4f} loss: {loss:.4f}")
                
                scheduler.step(total_loss)
                # Check if the learning rate was reduced
                if scheduler._last_lr[0] < optimizer.param_groups[0]['lr']:
                    accelerator.print(f"Learning rate reduced to: {scheduler._last_lr[0]}")
                
                ## log with tb
                timestamp = time.time()
                # stats_metrics.update({
                #     'Eval/loss':total_loss,
                #     'Eval/psnr':total_psnr
                # })
                stats_metrics = {
                    'Eval/loss':total_loss,
                    'Eval/psnr':total_psnr,
                    'Eval/lr': optimizer.param_groups[0]['lr']  # Log the learning rate
                }
                # stats_dict.update({
                #     'Eval/loss':total_loss.item(),
                #     'Eval/psnr':total_psnr.item()
                # })
                stats_dict = {
                    'Eval/loss':total_loss.item(),
                    'Eval/psnr':total_psnr.item(),
                    'Eval/lr': optimizer.param_groups[0]['lr']  # Log the learning rate
                }
                
                if stats_jsonl is not None:
                    fields = dict(stats_dict, timestamp=timestamp)
                    stats_jsonl.write(json.dumps(fields) + '\n')
                    stats_jsonl.flush()
                if stats_tfevents is not None:
                    global_step = int(cur_nimg / 10)
                    walltime = timestamp - start_time
                    for name, value in stats_metrics.items():
                        stats_tfevents.add_scalar(f'{name}', value, global_step=global_step, walltime=walltime)
                    stats_tfevents.flush()
                    # print("tf log sucessful!")
                
                stats_metrics = dict()
                stats_dict = dict() # clear dict entries for logging training
              


if __name__ == "__main__":
    main()
