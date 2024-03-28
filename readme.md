
## Zero-1-to-G: Single Stage 3D Generation with Splatter Images


### Install

```bash
# xformers is required! please refer to https://github.com/facebookresearch/xformers for details.
# for example, we use torch 2.1.0 + cuda 18.1
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -U xformers --index-url https://download.pytorch.org/whl/cu118

# a modified gaussian splatting (+ depth, alpha rendering)
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# for mesh extraction
pip install git+https://github.com/NVlabs/nvdiffrast

# other dependencies
pip install -r requirements.txt
```

### Training
First, please specify the path to your training data, which consists of splatter image gt and the multiview rendering gt:
```bash
DATA_DIR_BATCH_RENDERING='/path/to/your/rendering' # which contains the folders named by scene name
DATA_DIR_BATCH_SPLATTER_GT_ROOT='/path/to/your/rendering/splatter_gt' 
```

Then begin DDP training using python. Please adjust `acc_configs/gpu4.yaml` according to your available training GPUs.
```bash
### stage 1: train an auto-decoder to decode latent code to splatter image space

accelerate launch --main_process_port 29510 --config_file acc_configs/gpu4.yaml main_zero123plus_v4_batch_code.py big --workspace runs/zerp123plus_batch/workspace_ablation \
    --lr 1e-4 --num_epochs 20001 --eval_iter 20 --save_iter 20 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
    --prob_cam_jitter 0 --prob_grid_distortion 0 --input_size 320 --num_input_views 6 --num_views 26 \
    --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
    --desc 'ablation_2dot1_fixed_encode_range_4gpus-mix_diffusion_interval_10-resume20240315' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
    --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
    --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
    --model_type Zero123PlusGaussianCode \
    --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
    --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder --mix_diffusion_interval 10


### stage 2: finetune diffusion UNet to adapt to the splatter image decoder
### please specify the ckpt of trained auto-decoder using "--resume"

accelerate launch --config_file acc_configs/gpu4.yaml main_zero123plus_v4_batch_code_unet.py big --workspace runs/zerp123plus_batch/workspace_ablation \
    --num_epochs 10001 --eval_iter 20 --save_iter 20 --lr_scheduler Plat \
    --lr 2e-6 --min_lr_scheduled 1e-10 --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
    --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
    --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 0 \
    --desc 'ablation4_unet_fixed_encode_range-4gpus-resumeunet20240320_ep140' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
    --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
    --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
    --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCodeUnet \
    --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
    --decoder_upblocks_interpolate_mode "last_layer" \
    --resume "path/to/your/trained/auto-decoder"
```
where different training settings is provided in the script.

### Inference

```bash
### inference on the trained ckpt, which is specified by argument --resume.

accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference.py big --workspace runs/zerp123plus_batch/workspace_debug \
    --lr 2e-4 --num_epochs 10001 --eval_iter 10 --save_iter 10 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
    --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
    --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 0 \
    --desc 'debug_encode_splatter' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
    --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
    --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
    --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
    --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
    --decoder_upblocks_interpolate_mode "last_layer" \
    --codes_from_diffusion \
    --resume "path/to/your/finetuned-unet"
```

### Acknowledgement

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [nvdiffrast](https://github.com/NVlabs/nvdiffrast)
- [dearpygui](https://github.com/hoffstadt/DearPyGui)
- [tyro](https://github.com/brentyi/tyro)
- [lgm](https://github.com/3DTopia/LGM)

