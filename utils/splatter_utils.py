import torch
import os
import einops
from pytorch3d.transforms import quaternion_to_axis_angle, axis_angle_to_quaternion

# process the loaded splatters into 3-channel images
gt_attr_keys = ['pos', 'opacity', 'scale', 'rotation', 'rgbs']
start_indices = [0, 3, 4, 7, 11]
end_indices = [3, 4, 7, 11, 14]
attr_map = {key: (si, ei) for key, si, ei in zip (gt_attr_keys, start_indices, end_indices)}

### 2DGS
ordered_attr_list = ["pos", # 0-3
                'opacity', # 3-4
                'scale', # 4-7
                "rotation", # 7-11
                "rgbs", # 11-14
            ] # must be an ordered list according to the channels

sp_min_max_dict = {
    "pos": (-0.7, 0.7), 
    "scale": (-10., -2.),
    "rotation": (-5., 5.) #  (-6., 6.)
    }


# def fuse_splatters(splatters):
#     # fuse splatters
#     B, V, C, H, W = splatters.shape
    
#     x = splatters.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)
    
#     # # SINGLE VIEW splatter 
#     # x = splatters.permute(0, 1, 3, 4, 2)[:,0].reshape(B, -1, 14)
#     return x

# def load_splatter_mv_ply_as_dict(splatter_dir, device="cpu", range_01=True, use_2dgs=True, selected_attr_list=None, return_gassians=False):
    
#     splatter_mv = torch.load(os.path.join(splatter_dir, "splatters_mv.pt"), map_location='cpu').detach().cpu()
        
#     # print("\nLoading splatters_mv:", splatter_mv.shape) # [1, 14, 384, 256]

#     splatter_3Channel_image = {}
#     if return_gassians:
        
#         splatter_3Channel_image["gaussians_gt"] = splatter_mv.reshape(14, -1).permute(1,0)
    
#     if selected_attr_list is None:
#         selected_attr_list = ordered_attr_list
#     # print("selected_attr_list:", selected_attr_list)
            
#     for attr_to_encode in selected_attr_list:
#         si, ei = attr_map[attr_to_encode]
        
#         sp_image = splatter_mv[si:ei]
#         # print(f"{attr_to_encode}: {sp_image.min(), sp_image.max()}")

#         #  map to 0,1
#         if attr_to_encode == "pos":
#             sp_min, sp_max = sp_min_max_dict[attr_to_encode]
#             sp_image = (sp_image - sp_min)/(sp_max - sp_min)
#         elif attr_to_encode == "opacity":
#             sp_image = sp_image.repeat(3,1,1)
#         elif attr_to_encode == "scale":
#             sp_image = torch.log(sp_image)
#             sp_min, sp_max = sp_min_max_dict[attr_to_encode]
#             sp_image = (sp_image - sp_min)/(sp_max - sp_min)
#             if use_2dgs:
#                 # print("0 the first dim of scale: ", sp_image.shape, sp_image[0].min(), sp_image[0].max())
#                 sp_image[0]*=0
#                 # print("[after] 0 the first dim of scale: ", sp_image.shape, sp_image[0].min(), sp_image[0].max())
#         elif  attr_to_encode == "rotation":
#             assert (ei - si) == 4
            
#             quat = einops.rearrange(sp_image, 'c h w -> h w c')
#             axis_angle = quaternion_to_axis_angle(quat)
#             sp_image = einops.rearrange(axis_angle, 'h w c -> c h w')
#             sp_min, sp_max = sp_min_max_dict[attr_to_encode]
#             # print("rotation:", sp_image.view(3,-1).min(dim=1), sp_image.view(3,-1).max(dim=1))
#             sp_image = (sp_image - sp_min)/(sp_max - sp_min)
            
#         elif attr_to_encode == "rgbs":
#             # print("rgbs(utils)", sp_image.min(), sp_image.max())
#             pass
        
#         if range_01:
#             sp_image = sp_image.clip(0,1)
#         else:
#             # map to [-1,1]
#             sp_image = sp_image * 2 - 1
#             sp_image = sp_image.clip(-1,1)
        
#         # print(f"{attr_to_encode}: {sp_image.min(), sp_image.max(), sp_image.shape}")
#         assert sp_image.shape[0] == 3
#         splatter_3Channel_image[attr_to_encode] = sp_image.detach().cpu()
    
#     return splatter_3Channel_image


def load_splatter_mv_ply_as_dict(splatter_dir, device="cpu", range_01=True, use_2dgs=True, selected_attr_list=None, return_gassians=False):
    
    splatter_mv = torch.load(os.path.join(splatter_dir, "splatters_mv.pt"), map_location='cpu').detach().cpu()
        
    # print("\nLoading splatters_mv:", splatter_mv.shape) # [1, 14, 384, 256]

    splatter_3Channel_image = {}
    if return_gassians:
        
        splatter_3Channel_image["gaussians_gt"] = splatter_mv.reshape(14, -1).permute(1,0)
    
    if selected_attr_list is None:
        selected_attr_list = ordered_attr_list
    # print("selected_attr_list:", selected_attr_list)
            
    for attr_to_encode in selected_attr_list:
        si, ei = attr_map[attr_to_encode]
        
        sp_image = splatter_mv[si:ei]
        # print(f"{attr_to_encode}: {sp_image.min(), sp_image.max()}")

        #  map to 0,1
        if attr_to_encode == "pos":
            sp_min, sp_max = sp_min_max_dict[attr_to_encode]
            sp_image = (sp_image - sp_min)/(sp_max - sp_min)
        elif attr_to_encode == "opacity":
            sp_image = sp_image.repeat(3,1,1)
        elif attr_to_encode == "scale":
            sp_image = torch.log(sp_image)
            sp_min, sp_max = sp_min_max_dict[attr_to_encode]
            sp_image = (sp_image - sp_min)/(sp_max - sp_min)
            if use_2dgs:
                sp_image[0]*=0
        elif  attr_to_encode == "rotation":
            assert (ei - si) == 4
            
            quat = einops.rearrange(sp_image, 'c h w -> h w c')
            axis_angle = quaternion_to_axis_angle(quat)
            sp_image = einops.rearrange(axis_angle, 'h w c -> c h w')
            sp_min, sp_max = sp_min_max_dict[attr_to_encode]
            # print("rotation:", sp_image.view(3,-1).min(dim=1), sp_image.view(3,-1).max(dim=1))
            sp_image = (sp_image - sp_min)/(sp_max - sp_min)
            
        elif attr_to_encode == "rgbs":
            # print("rgbs(utils)", sp_image.min(), sp_image.max())
            pass
        
        if range_01:
            sp_image = sp_image.clip(0,1)
        else:
            # map to [-1,1]
            sp_image = sp_image * 2 - 1
            sp_image = sp_image.clip(-1,1)
        
        # print(f"{attr_to_encode}: {sp_image.min(), sp_image.max(), sp_image.shape}")
        assert sp_image.shape[0] == 3
        splatter_3Channel_image[attr_to_encode] = sp_image.detach().cpu()
    
    return splatter_3Channel_image