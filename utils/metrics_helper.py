# from torch_fidelity import calculate_statistics, calculate_metrics
import os
import numpy as np
import torch

from ipdb import set_trace as st
import einops
from PIL import Image

def save_real_image_statistics(dataset, num_samples=100, file_path='real_image_statistics.npz'):
    real_images = []
    for idx, data in enumerate(dataset):
        img = data['images_output']  # Add batch dimension
        st() # normalize to 0-255?
        real_images.append(img)
    real_images = torch.cat(real_images, dim=0)  # [N, C, H, W]
    st()
    real_stats = calculate_statistics(real_images, cuda=True)
    np.savez(file_path, mu=real_stats['mu'], sigma=real_stats['sigma'], num_images=real_images.shape[0])
    print(f"Saved real image statistics to {file_path}")

def load_real_image_statistics(file_path='real_image_statistics.npz'):
    stats = np.load(file_path)
    return {'mu': stats['mu'], 'sigma': stats['sigma'], 'num_images': stats['num_images']}


def save_real_image_npz(dataset, num_samples=100, file_path='real_image_statistics.npz'):
    real_images = []
    for idx, data in enumerate(dataset):
        img = data['images_output']  # Add batch dimension
        real_images.append(img[:,0:1])
    real_images = torch.cat(real_images, dim=0)  # [N, V, C, H, W], 0-1

    real_images = einops.rearrange(real_images, "N V C H W -> (N V) H W C") # arr_0: (10000, 256, 256, 3)
    real_images_arr = (real_images * 255).cpu().numpy().astype(np.uint8) # 0-255, uint8 
    
    np.savez(file_path, arr_0 = real_images_arr)# ['mu', 'sigma', 'mu_s', 'sigma_s', 'mu_clip', 'sigma_clip', 'arr_0']
    print(f"Saved {real_images_arr.shape[0]} real image statistics to {file_path}")


def save_generated_image_npz(real_images, file_path='generated_image_statistics.npz'):
    real_images = torch.cat(real_images, dim=0)  # [N, V, C, H, W], 0-1

    real_images = einops.rearrange(real_images, "N V C H W -> (N V) H W C") # arr_0: (10000, 256, 256, 3)
    real_images_arr = (real_images * 255).cpu().numpy().astype(np.uint8) # 0-255, uint8 
    
    np.savez(file_path, arr_0 = real_images_arr)# ['mu', 'sigma', 'mu_s', 'sigma_s', 'mu_clip', 'sigma_clip', 'arr_0']
    print(f"Saved {real_images_arr.shape[0]} real image statistics to {file_path}")

def save_generated_image_png(real_images, idx, file_path='generated_image_statistics.npz'):
    
    real_images = einops.rearrange(real_images, "N V C H W -> (N V) H W C") # arr_0: (10000, 256, 256, 3)
    real_images_arr = (real_images * 255).cpu().numpy().astype(np.uint8) # 0-255, uint8 
    for i, _img in enumerate(real_images_arr):
        _path = f"{file_path}/{idx}_{i}.png"
        assert not os.path.exists(_path)
        Image.fromarray(_img).save(_path)
    print(f"Saved all {i+1} views for object {idx}")
