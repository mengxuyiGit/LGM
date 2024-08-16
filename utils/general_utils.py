
import torch
import sys
from datetime import datetime
import numpy as np
import random
import matplotlib.pyplot as plt

# def colormap(img, cmap='jet'):
#     import matplotlib.pyplot as plt
#     W, H = img.shape[:2]
#     dpi = 300
#     fig, ax = plt.subplots(1, figsize=(H/dpi, W/dpi), dpi=dpi)
#     im = ax.imshow(img, cmap=cmap)
#     ax.set_axis_off()
#     fig.colorbar(im, ax=ax)
#     fig.tight_layout()
#     fig.canvas.draw()
#     data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#     img = torch.from_numpy(data / 255.).float().permute(2,0,1)
#     plt.close()
#     return img


def colormap(batch_img, cmap='jet', colorbar_shrink=0.2, font_size=3):
    n, W, H = batch_img.shape[:3]
    dpi = 300
    batch_colored = []
    
    for i in range(n):
        img = batch_img[i]
        fig, ax = plt.subplots(1, figsize=(H/dpi, W/dpi), dpi=dpi)
        im = ax.imshow(img, cmap=cmap)
        ax.set_axis_off()
        # fig.colorbar(im, ax=ax)
        # fig.colorbar(im, ax=ax, shrink=colorbar_shrink)
        # Adjust the colorbar size and font size
        cbar = fig.colorbar(im, ax=ax, shrink=colorbar_shrink)
        cbar.ax.tick_params(labelsize=font_size)  # Set font size for colorbar labels
        
        fig.tight_layout()
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img_colored = torch.from_numpy(data / 255.).float().permute(2, 0, 1)
        batch_colored.append(img_colored)
        plt.close(fig)
    
    return torch.stack(batch_colored)

# def colormap(batch_img, cmap='jet', colorbar_shrink=0.2):
#     n, W, H = batch_img.shape[:3]
#     dpi = 300
#     batch_colored = []
    
#     for i in range(n):
#         img = batch_img[i]
#         fig, ax = plt.subplots(1, figsize=(H/dpi, W/dpi), dpi=dpi)
#         im = ax.imshow(img, cmap=cmap)
#         ax.set_axis_off()
#         # Adjust the colorbar size with the shrink parameter
#         fig.colorbar(im, ax=ax, shrink=colorbar_shrink)
#         fig.tight_layout()
#         fig.canvas.draw()
#         data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#         data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#         img_colored = torch.from_numpy(data / 255.).float().permute(2, 0, 1)
#         batch_colored.append(img_colored)
#         plt.close(fig)
    
#     return torch.stack(batch_colored)