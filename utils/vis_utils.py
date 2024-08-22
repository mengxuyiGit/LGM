import torch
import os
import einops
import kiui
import numpy as np

from torchvision import transforms
from PIL import Image
from pytorch3d.transforms import quaternion_to_axis_angle, axis_angle_to_quaternion


# process the loaded splatters into 3-channel images
gt_attr_keys = ['pos', 'opacity', 'scale', 'rotation', 'rgbs']
start_indices = [0, 3, 4, 7, 11]
end_indices = [3, 4, 7, 11, 14]
attr_map = {key: (si, ei) for key, si, ei in zip (gt_attr_keys, start_indices, end_indices)}

ordered_attr_list = ["pos", # 0-3
                'opacity', # 3-4
                'scale', # 4-7
                "rotation", # 7-11
                "rgbs", # 11-14
            ] # must be an ordered list according to the channels

sp_min_max_dict = {
    "pos": (-0.7, 0.7), 
    "scale": (-10., -2.),
    "rotation": (-3., 3.)
    }

def load_splatter_mv_ply_as_dict(splatter_mv, device="cpu"):
    
    # # splatter_mv = torch.load(os.path.join(splatter_dir, "splatters_mv.pt")).detach().cpu() # [14, 384, 256]
    # splatter_mv = torch.load(os.path.join(splatter_dir, "splatters_mv.pt"), map_location='cpu').detach().cpu()

        
    splatter_3Channel_image = {}
            
    for attr_to_encode in ordered_attr_list:
        # print("latents_all_attr_list <-",attr_to_encode)
        si, ei = attr_map[attr_to_encode]
        
        sp_image = splatter_mv[si:ei]
        # print(f"{attr_to_encode}: {sp_image.min(), sp_image.max()}")

        #  map to 0,1
        if attr_to_encode in ["pos"]:
            sp_min, sp_max = sp_min_max_dict[attr_to_encode]
            sp_image = (sp_image - sp_min)/(sp_max - sp_min)
        elif attr_to_encode == "opacity":
            sp_image = sp_image.repeat(3,1,1)
        elif attr_to_encode == "scale":
            sp_image = torch.log(sp_image)
            sp_min, sp_max = sp_min_max_dict[attr_to_encode]
            sp_image = (sp_image - sp_min)/(sp_max - sp_min)
            sp_image = sp_image.clip(0,1)
        elif  attr_to_encode == "rotation":
            assert (ei - si) == 4
            quat = einops.rearrange(sp_image, 'c h w -> h w c')
            axis_angle = quaternion_to_axis_angle(quat)
            sp_image = einops.rearrange(axis_angle, 'h w c -> c h w')
            # print(f"{attr_to_encode}: {sp_image.min(), sp_image.max()}")
            sp_min, sp_max = sp_min_max_dict[attr_to_encode]
            sp_image = (sp_image - sp_min)/(sp_max - sp_min)
        elif attr_to_encode == "rgbs":
            pass
        
        # map to [-1,1]
        sp_image = sp_image * 2 - 1
        sp_image = sp_image.clip(-1,1)
        
        # print(f"{attr_to_encode}: {sp_image.min(), sp_image.max(), sp_image.shape}")
        assert sp_image.shape[0] == 3
        splatter_3Channel_image[attr_to_encode] = sp_image.detach().cpu()
    
    
    splatter_3Channel_image = torch.stack([splatter_3Channel_image[attr] for attr in ordered_attr_list], dim=0)
    
    return splatter_3Channel_image


def save_splatter_vis(out, path, opt):
    batch_splatter_mv=  einops.rearrange(out['gaussians'], 'b (v h w) c -> b v c h w', v=6, h=opt.splat_size, w=opt.splat_size)
    batch_splatter_mv = einops.rearrange(batch_splatter_mv, 'b (m n) c h w -> b c (m h) (n w)', m=3, n=2)
    batch_splatter_mv_vis = []
    for _splatter_mv in batch_splatter_mv: # [v c h w]
        splatter_image_vis = load_splatter_mv_ply_as_dict(_splatter_mv)
        # save vis
        images_to_save = splatter_image_vis.detach().cpu().numpy() # [5, 3, output_size, output_size]
        images_to_save = (images_to_save + 1) * 0.5
        images_to_save = einops.rearrange(images_to_save, "a c (m h) (n w) -> (a h) (m n w) c", m=3, n=2)
        
        # kiui.write_image(f'{opt.workspace}/images_batch_attr_Lencode_Rdecoded_{epoch}_{i}.jpg', images_to_save)
        batch_splatter_mv_vis.append(images_to_save)
        
    batch_splatter_mv_vis = np.concatenate(batch_splatter_mv_vis, axis=0)
    kiui.write_image(path, batch_splatter_mv_vis)
    
    return None