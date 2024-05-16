DATA_DIR_BATCH_RENDERING='/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999'
DATA_DIR_BATCH_SPLATTER_GT_ROOT='/home/xuyimeng/Repo/LGM/runs/splatter_gt'
DATA_DIR_BATCH_VAE_SPLATTER_ROOT="/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold/workspace_optimize"


# [TRAIN diffusion unet - use rendering loss]
# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_unet_rendering_loss.py big --workspace runs/marigold_unet/workspace_ovft \
#     --lr 1e-4 --num_epochs 10001 --eval_iter 20 --save_iter 100 --lr_scheduler Plat \
#     --lr_scheduler_patience 5 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'marigold-multichannel-unet-divide-factor_in-zero-latent-loss-only-learn-rgb-fix-t=10' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --overfit_one_scene --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnet --data_path_vae_splatter ${DATA_DIR_BATCH_VAE_SPLATTER_ROOT} \
#     --custom_pipeline "./zero123plus/pipeline_v4.py" --render_input_views --attr_group_mode "v5"
#     #  --plot_attribute_histgram 'scale' 

# # [APR 29: use the original 4 channel unet]
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_unet_rendering_loss.py big --workspace runs/marigold_unet/workspace_ovft \
#     --lr 1e-4 --num_epochs 10001 --eval_iter 20 --save_iter 100 --lr_scheduler Plat \
#     --lr_scheduler_patience 5 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'marigold-unet-WITH-latent-loss-only-learn-rgb-v2-RANDOM-t-MAX' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --overfit_one_scene --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnet --data_path_vae_splatter ${DATA_DIR_BATCH_VAE_SPLATTER_ROOT} \
#     --custom_pipeline "./zero123plus/pipeline_v2.py" --render_input_views --attr_group_mode "v5"
#     #  --plot_attribute_histgram 'scale' 

# # [APR 30: rgb OK, try other attributes without additional condition]
# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_unet_rendering_loss.py big --workspace runs/marigold_unet/workspace_debug \
#     --lr 1e-4 --num_epochs 10001 --eval_iter 50 --save_iter 100 --lr_scheduler Plat \
#     --lr_scheduler_patience 5 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'marigold-unet-WITH-latent-loss-only-learn-XYZ-v2-RANDOM-t-MAX' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --overfit_one_scene --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnet --data_path_vae_splatter ${DATA_DIR_BATCH_VAE_SPLATTER_ROOT} \
#     --custom_pipeline "./zero123plus/pipeline_v2.py" --render_input_views --attr_group_mode "v5" \
#     --lambda_latent 1 --attr_to_learn "xyz"
#     # --fixed_noise_level 500


# # [APR 30: use 4 channel unet with cross domain attention and domain condition]
# # export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_unet_rendering_loss.py big --workspace runs/marigold_unet/workspace_CD_ovft \
#     --lr 1e-4 --num_epochs 10001 --eval_iter 20 --save_iter 100 --lr_scheduler Plat \
#     --lr_scheduler_patience 5 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'marigold-unet-render-gt-latents' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain --data_path_vae_splatter ${DATA_DIR_BATCH_VAE_SPLATTER_ROOT} \
#     --custom_pipeline "./zero123plus/pipeline_v5.py" --render_input_views --attr_group_mode "v5" \
#     --overfit_one_scene

# # [MAY 02: train a single attribute on all scenes]
# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --main_process_port 29513 --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_unet_rendering_loss.py big --workspace runs/marigold_unet/workspace_CD_debug \
#     --lr 1e-4 --num_epochs 10001 --eval_iter 10 --save_iter 10 --lr_scheduler Plat \
#     --lr_scheduler_patience 5 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'marigold-unet-v-pred-cd-spatial-cat-all-data' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain --data_path_vae_splatter ${DATA_DIR_BATCH_VAE_SPLATTER_ROOT} \
#     --custom_pipeline "./zero123plus/pipeline_v5.py" --render_input_views --attr_group_mode "v5" \
#     --cd_spatial_concat --skip_training 

#     # --attr_to_learn "rgbs" 
#     # --overfit_one_scene 

# # [MAY 05: train spatial cat on all scenes]
# # export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --main_process_port 29515 --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_unet_rendering_loss.py big --workspace runs/marigold_unet/workspace_CD_train \
#     --lr 1e-4 --num_epochs 10001 --eval_iter 10 --save_iter 10 --lr_scheduler Plat \
#     --lr_scheduler_patience 5 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc '4-gpus-marigold-unet-v-pred-cd-spatial-cat-all-data' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain --data_path_vae_splatter ${DATA_DIR_BATCH_VAE_SPLATTER_ROOT} \
#     --custom_pipeline "./zero123plus/pipeline_v5.py" --render_input_views --attr_group_mode "v5" \
#     --cd_spatial_concat

# # [MAY 02: overfit single scene, all attributes, with cross domain attention]
# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_unet_rendering_loss.py big --workspace runs/marigold_unet/workspace_CD_debug \
#     --lr 1e-4 --num_epochs 10001 --eval_iter 20 --save_iter 100 --lr_scheduler Plat \
#     --lr_scheduler_patience 5 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'marigold-unet-single-attr-rgbs-train' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain --data_path_vae_splatter ${DATA_DIR_BATCH_VAE_SPLATTER_ROOT} \
#     --custom_pipeline "./zero123plus/pipeline_v5_dev.py" --render_input_views --attr_group_mode "v5" \
#     --overfit_one_scene 

# # [MAY 06: overfit single scene, all attributes, with cross domain attention]
# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_unet_rendering_loss.py big --workspace runs/marigold_unet/workspace_CD_debug \
#     --lr 1e-4 --num_epochs 10001 --eval_iter 20 --save_iter 100 --lr_scheduler Plat \
#     --lr_scheduler_patience 5 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'marigold-unet-single-attr-rgbs-train' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain --data_path_vae_splatter ${DATA_DIR_BATCH_VAE_SPLATTER_ROOT} \
#     --custom_pipeline "./zero123plus/pipeline_v6_set.py" --render_input_views --attr_group_mode "v5" \
#     --overfit_one_scene 

# [MAY 06: overfit single scene, all attributes, with cross domain attention]
# export CUDA_VISIBLE_DEVICES=1
accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_unet_rendering_loss.py big --workspace runs/marigold_unet/workspace_CD_debug \
    --lr 1e-4 --num_epochs 10001 --eval_iter 100 --save_iter 200 --lr_scheduler Plat \
    --lr_scheduler_patience 5 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
    --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 7 \
    --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
    --desc 'debug_bsz2_dev_cleanCodeForZero1toG' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
    --set_random_seed --batch_size 2 --num_workers 1 \
    --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
    --scale_clamp_max -2 --scale_clamp_min -10 \
    --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
    --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
    --model_type Zero123PlusGaussianMarigoldUnetCrossDomain --data_path_vae_splatter ${DATA_DIR_BATCH_VAE_SPLATTER_ROOT} \
    --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
    --bg 1.0 --only_train_attention --rendering_loss_use_weight_t 
    # --fixed_noise_level 300
    # --rendering_loss_use_weight_t  --overfit_one_scene 

# # [MAY 15] train all scenes with constant large lr
# accelerate launch --main_process_port 29514 --config_file acc_configs/gpu8.yaml main_zero123plus_v5_batch_marigold_unet_rendering_loss.py big --workspace runs/marigold_unet/workspace_CD_train \
#     --lr 1e-4 --num_epochs 10001 --eval_iter 20 --save_iter 20 --lr_scheduler Plat \
#     --lr_scheduler_patience 5 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'LARGE-LR-Attn-only_attn-rendering_w_t' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain --data_path_vae_splatter ${DATA_DIR_BATCH_VAE_SPLATTER_ROOT} \
#     --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --only_train_attention --rendering_loss_use_weight_t


# # 4gpus: 
# export CUDA_VISIBLE_DEVICES=0,1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_unet_rendering_loss.py big --workspace runs/marigold_unet/workspace_CD_debug \
#     --lr 1e-4 --num_epochs 10001 --eval_iter 10 --save_iter 10 --lr_scheduler Plat \
#     --lr_scheduler_patience 5 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'marigold-unet-v7-seq-reshape-v-pred-loss-white' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain --data_path_vae_splatter ${DATA_DIR_BATCH_VAE_SPLATTER_ROOT} \
#     --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0

# # [MAY 06]: only train transformer
# accelerate launch --main_process_port 29515 --config_file acc_configs/gpu4.yaml main_zero123plus_v5_batch_marigold_unet_rendering_loss.py big --workspace runs/marigold_unet/workspace_CD_train \
#     --lr 1e-4 --num_epochs 10001 --eval_iter 10 --save_iter 10 --lr_scheduler Plat \
#     --lr_scheduler_patience 5 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'marigold-unet-v7_seq-only-train-attn-v-pred-white-bg' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain --data_path_vae_splatter ${DATA_DIR_BATCH_VAE_SPLATTER_ROOT} \
#     --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --only_train_attention


# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_unet_rendering_loss.py big --workspace runs/marigold_unet/workspace_debug \
#     --lr 1e-5 --num_epochs 10001 --eval_iter 100 --save_iter 100 --lr_scheduler Plat \
#     --lr_scheduler_patience 5 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 10 \
#     --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'marigold-unet-w-rendering-loss' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --overfit_one_scene --codes_from_encoder \
#     --model_type Zero123PlusGaussianCodeUnet --data_path_vae_splatter ${DATA_DIR_BATCH_VAE_SPLATTER_ROOT}