import tyro
import time
import random

import torch
from core.options import AllConfigs
from core.models import LGM
from accelerate import Accelerator, DistributedDataParallelKwargs
from safetensors.torch import load_file

import kiui
from diffusers import DiffusionPipeline, DDPMScheduler, EulerAncestralDiscreteScheduler
from ipdb import set_trace as st

import json
from datetime import datetime
import os
from tqdm import tqdm
from core.dataset_v5_marigold import ordered_attr_list


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
        print(f"name: {name}")
        if param.requires_grad:
        # if True:
            # Check if the current parameter is different from the initial
            if not torch.equal(initial_weights[name], param.data):
                print(f"Weight updated: {name}")
                updated = True
        else:
            print(f"{name} does not requires grad")
    if not updated:
        print("No weights were updated.")


def scale_image(image):
    image = image * 0.5 / 0.8
    return image


def save_stats(mean, std, filename):
    # Convert tensors to lists for JSON serialization
    mean_list = mean.tolist()
    std_list = std.tolist()
    
    # Create a dictionary to store the stats
    stats = {
        'mean': mean_list,
        'std': std_list
    }
    
    # Write the dictionary to a JSON file
    with open(filename, 'w') as f:
        json.dump(stats, f)

def load_stats(filename):
    # Read the JSON file
    with open(filename, 'r') as f:
        stats = json.load(f)
    
    # Convert lists back to tensors
    mean = torch.tensor(stats['mean'])
    std = torch.tensor(stats['std'])
    
    return mean, std


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
    
    from core.dataset_v5_marigold_stats import ObjaverseDataset as Dataset

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

    # load sd/svd 
    # Load zero123plus model
    import sys
    sys.path.append('./zero123plus')

    pipe = DiffusionPipeline.from_pretrained(
        opt.model_path,
        custom_pipeline=opt.custom_pipeline
    ).to('cuda')
    pipe.vae.requires_grad_(False).eval()
    
    # #########
             
    # for attr_to_encode in ordered_attr_list_local:
    #     sp_image = data[attr_to_encode]
    #     # print(f"[data]{attr_to_encode}: {sp_image.min(), sp_image.max()}")
    #     images_all_attr_list.append(sp_image)
    # images_all_attr_batch = torch.stack(images_all_attr_list)

    # A, B, _, _, _ = images_all_attr_batch.shape # [5, 1, 3, 384, 256]
    # images_all_attr_batch = einops.rearrange(images_all_attr_batch, "A B C H W -> (B A) C H W")
    
    # if save_path is not None:    
    #     images_to_save = images_all_attr_batch.detach().cpu().numpy() # [5, 3, output_size, output_size]
    #     images_to_save = (images_to_save + 1) * 0.5
    #     images_to_save = einops.rearrange(images_to_save, "a c (m h) (n w) -> (a h) (m n w) c", m=3, n=2)

    # # do vae.encode
    # sp_image_batch = scale_image(images_all_attr_batch)
    # sp_image_batch = self.pipe.vae.encode(sp_image_batch).latent_dist.sample() * self.pipe.vae.config.scaling_factor
    # latents_all_attr_encoded = scale_latents(sp_image_batch) # torch.Size([5,
    
    # #########

    # os.make dir
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    opt.workspace = os.path.join(opt.workspace, f"{time_str}-{opt.train_unet_single_attr}")
    os.makedirs(opt.workspace, exist_ok=True)

  
    # loop
    with torch.no_grad():
        # for epoch in range(opt.num_epochs):
        # attr = "pos"
        # for attr in ordered_attr_list:
        for attr in opt.train_unet_single_attr:
            # train
           
            model.train()
            total_loss = 0
            total_psnr = 0
            
            print(f"Calculating stats for {attr}")
            attr_latents = []
            for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), disable=(opt.verbose_main), desc = f"Calculating stats: {attr}"):
            # for i, data in enumerate(train_dataloader):
                
                # print("data: ", data.keys())

                # # make sure the image is within [-1,1]
                # print(attr, data[attr].min(), data[attr].max(), data[attr].shape)
                
                # scale image
                scaled_img = scale_image(data[attr])
                
                # vae.encode
                _latent = pipe.vae.encode(scaled_img).latent_dist.sample() * pipe.vae.config.scaling_factor
                
                # calculate latent mean and std (this stats are before the scale_latents)
                attr_latents.append(_latent.detach().cpu())
                
                continue
            
            
                with accelerator.accumulate(model):

                    optimizer.zero_grad()

                    step_ratio = (epoch + i / len(train_dataloader)) / opt.num_epochs

                    out = model(data, step_ratio)
                    loss = out['loss']
                    psnr = out['psnr']
                    accelerator.backward(loss)

                    # gradient clipping
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)

                    optimizer.step()
                    scheduler.step()

                    total_loss += loss.detach()
                    total_psnr += psnr.detach()
            
            # calculate mean and std
            latents = torch.cat(attr_latents, dim=0)
            mean = torch.mean(latents, dim=0)
            std = torch.std(latents, dim=0)
            print(latents.shape)
        
            fname = f"{opt.workspace}/{attr}_stats.txt"
            save_stats(mean=mean, std=std, filename=fname)
            loaded_mean, loaded_std = load_stats(fname)
           
            print(f"mean: {mean.shape} \t loaded_mean: {loaded_mean.shape}")
            print(f"std: {std.shape} \t loaded_std: {loaded_std.shape}")
            assert torch.allclose(mean, loaded_mean)
            assert torch.allclose(std, loaded_std)
            print("save and load are equal!!")
            


if __name__ == "__main__":
    main()
