import os
from skimage.metrics import structural_similarity as ssim
from skimage import io
from PIL import Image
import numpy as np
import torch
import einops
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from datetime import datetime
from kiui.lpips import LPIPS

# Initialize LPIPS loss
self_lpips_loss = LPIPS(net='vgg')
self_lpips_loss.requires_grad_(False)

# Metric calculation functions
def calculate_ssim(gt_images, pred_images):
    gt_images = F.interpolate(gt_images, (320, 320), mode='bilinear', align_corners=False) 
    pred_images = F.interpolate(pred_images, (320, 320), mode='bilinear', align_corners=False)
    
    gt_images, pred_images = np.asarray(gt_images), np.asarray(pred_images)

    ssim_batch = []
    for img1, img2 in zip(gt_images, pred_images):
        ssim_batch.append(ssim(img1, img2, multichannel=True, channel_axis=0, data_range=1))
    
    ssim_value = sum(ssim_batch) / len(ssim_batch)
    return ssim_value

def calculate_lpips(gt_images, pred_images):
    loss_lpips = self_lpips_loss(
        F.interpolate(gt_images * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
        F.interpolate(pred_images * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
    ).mean()
    return loss_lpips

def calculate_psnr(gt_images, pred_images):
    mse = F.mse_loss(gt_images, pred_images)
    psnr = 10 * torch.log10(1 / mse)
    return psnr

import glob
def get_image_lists(folder_name, gt_pattern, pred_pattern, suffix):
    # List all files in the folder
    print(folder_name)
    print("Pattern:", gt_pattern, pred_pattern)
    gt_images = sorted(glob.glob(f"{folder_name}/**/{gt_pattern}", recursive=True))
    pred_images = sorted(glob.glob(f"{folder_name}/**/{pred_pattern}", recursive=True))
    # print(gt_images)
    # print(pred_images)
    print("Total gt/pred found:", len(gt_images), len(pred_images))
    
    return gt_images, pred_images


# Main metric calculation function
def calculate_metrics(folder_name, metrics=['psnr', 'ssim', 'lpips'], gt_pattern='gt.jpg', pred_pattern='sample_cfg', suffix='jpg', image_size=512, num_views=16):
 
  
    gt_images, pred_images = get_image_lists(folder_name, gt_pattern, pred_pattern, suffix)
    
    
    total_psnr, total_ssim, total_lpips = [], [], []
    pred_file_names = []
    
    # Ensure that ground truth and predicted image files are paired correctly
    for gt_image, pred_image in zip(gt_images, pred_images):
        # print(f"Processing: {gt_image}, {pred_image}...")
        print(f".........: {os.path.basename(gt_image)}, {os.path.basename(pred_image)}...")
        pred_file_names.append(os.path.basename(pred_image))
        # continue
        
        # Load images
        gt_img = Image.open(os.path.join(folder_name, gt_image))
        pred_img = Image.open(os.path.join(folder_name, pred_image))
        
        # Convert to tensor
        gt_tensor = torch.tensor(np.asarray(gt_img)).to(torch.float32) / 255.
        pred_tensor = torch.tensor(np.asarray(pred_img)).to(torch.float32) / 255.

        # Reshape images if required (e.g., for multi-view)
        gt_tensor = einops.rearrange(gt_tensor, "h (n w) c -> n c h w", n=num_views)
        pred_tensor = einops.rearrange(pred_tensor, "h (n w) c -> n c h w", n=num_views)
        
        # Calculate metrics
        if 'psnr' in metrics:
            psnr = calculate_psnr(gt_tensor, pred_tensor)
            total_psnr.append(psnr)
        if 'ssim' in metrics:
            ssim_value = calculate_ssim(gt_tensor, pred_tensor)
            total_ssim.append(ssim_value)
        if 'lpips' in metrics:
            lpips_value = calculate_lpips(gt_tensor, pred_tensor)
            total_lpips.append(lpips_value)

    # Calculate averages and save to file
    metric_file_folder = f"{folder_name}/metrics"
    os.makedirs(metric_file_folder, exist_ok=True)
    
    if 'psnr' in metrics:
        avg_psnr = sum(total_psnr) / len(total_psnr)
        with open(f"{metric_file_folder}/metrics_psnr.txt", "a") as f:
            print("==============================", file=f)
            f.write(f"Average PSNR: {avg_psnr}\nDate: {datetime.now()}\n")
            print(f"Average PSNR: {avg_psnr}")
            
            # also save the individual PSNR values
            print("----------------------", file=f)
            for i, psnr in enumerate(total_psnr):
                f.write(f"{pred_file_names[i]}, psnr: {psnr}\n")
            print("----------------------\n\n", file=f)
        
        
    if 'ssim' in metrics:
        avg_ssim = sum(total_ssim) / len(total_ssim)
        with open(f"{metric_file_folder}/metrics_ssim.txt", "a") as f:
            print("==============================", file=f)
            f.write(f"Average SSIM: {avg_ssim}\nDate: {datetime.now()}\n")
            print(f"Average SSIM: {avg_ssim}")
            # also save the individual SSIM values
            print("----------------------", file=f)
            for i, ssim in enumerate(total_ssim):
                f.write(f"{pred_file_names[i]}, ssim: {ssim}\n")
            print("----------------------\n\n", file=f)
            
        
    if 'lpips' in metrics:
        avg_lpips = sum(total_lpips) / len(total_lpips)
        with open(f"{metric_file_folder}/metrics_lpips.txt", "a") as f:
            print("==============================", file=f)
            f.write(f"Average LPIPS: {avg_lpips}\nDate: {datetime.now()}\n")
            print(f"Average LPIPS: {avg_lpips}")
            # also save the individual LPIPS values
            print("----------------------", file=f)
            for i, lpips in enumerate(total_lpips):
                f.write(f"{pred_file_names[i]}, lpips: {lpips}\n")
            print("----------------------\n\n", file=f)
    
    print("Metrics calculation completed. Saved to:", metric_file_folder)

# CLI parser
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", type=str, required=True)
    parser.add_argument("--metrics", type=str, nargs='+', default=['psnr', 'ssim', 'lpips'])
    parser.add_argument("--gt_pattern", type=str, default='gt.jpg')
    parser.add_argument("--pred_pattern", type=str, default='sample_cfg')
    parser.add_argument("--suffix", type=str, default='jpg')
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_views", type=int, default=16)

    args = parser.parse_args()
    calculate_metrics(
        folder_name=args.folder_name,
        metrics=args.metrics,
        gt_pattern=args.gt_pattern,
        pred_pattern=args.pred_pattern,
        suffix=args.suffix,
        image_size=args.image_size,
        num_views=args.num_views
    )
