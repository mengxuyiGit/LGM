DATA_DIR_BATCH_RENDERING='/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999'
DATA_DIR_BATCH_SPLATTER_GT_ROOT='/home/xuyimeng/Repo/LGM/runs/splatter_gt'
# DATA_DIR_BATCH_SPLATTER_GT_ROOT_FIX_ROT='/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/LGM_optimize_splatter/optimize_fixrot/00001-batch-es10-Plat-patience_2-factor_0.5-eval_5-adamW-subset_0_5_splat128-inV6-lossV20-lr0.003/9000-9999'
# DATA_DIR_BATCH_SPLATTER_GT_ROOT_FIX_ROT="/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/LGM_optimize_splatter/optimize_fixrot/init-00001-batch-es10-Plat-patience_2-factor_0.5-eval_5-adamW-subset_0_5_splat128-inV6-lossV20-lr0.003/9000-9999"
DATA_DIR_BATCH_SPLATTER_GT_ROOT_FIX_ROT="/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/LGM_optimize_splatter/optimize_fixrot/00002-batch-es10-Plat-patience_2-factor_0.5-eval_5-adamW-subset_0_5_splat128-inV6-lossV20-lr0.003/9000-9999"

DATA_DIR_BATCH_RENDERING_SRN='/home/xuyimeng/Data/SRN/srn_cars/cars_train'
DATA_DIR_BATCH_SPLATTER_GT_ROOT_SRN_HQ='/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/LGM_optimize_splatter/optimize'

# python infer.py big --resume pretrained/model_fp16.safetensors --workspace workspace_test/0123 \
#     --test_path ${DATA_DIR_BATCH_RENDERING}/ffb0d644238b4c679658aa0ee46ac6da


# python infer_zero123plus.py big --resume pretrained/model_fp16.safetensors --workspace workspace_test/0123/srn_car_yellow_front \
#     --test_path ${DATA_DIR_BATCH_RENDERING}/ffb0d644238b4c679658aa0ee46ac6da \
#     --num_input_views 6 --model_type LGM


# python infer_cw.py big --resume pretrained/model_fp16.safetensors --workspace workspace_test/cw \
#     --test_path ${DATA_DIR_BATCH_RENDERING}/ffb0d644238b4c679658aa0ee46ac6da \
#     --num_input_views 6 --model_type LGM


# # [INFERENCE with diffusion]
# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference.py big --workspace runs/zerp123plus_batch/workspace_inference \
#     --lr 2e-4 --num_epochs 10001 --eval_iter 10 --save_iter 10 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 0 \
#     --desc 'ablation4_unet_20240320_settimesteps' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interva   l 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_diffusion --one_step_diffusion 1000 \
#     --resume "runs/zerp123plus_batch/workspace_ablation/20240320-ablation4_unet_fixed_encode_range-3gpus-resume_ep120-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0-lr4e-06-Plat100/eval_epoch_140/model.safetensors"
#     # --resume "runs/zerp123plus_batch/workspace_ablation/20240316-ablation4_unet_fixed_encode_range-4gpus-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0-lr1e-05-Plat100/eval_epoch_120/model.safetensors"


# # [INFERENCE ORIGIANL ZERO123plus with diffusion]
# export CUDA_VISIBLE_DEVICES=2
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference.py big --workspace runs/zerp123plus_batch/workspace_debug \
#     --lr 2e-4 --num_epochs 10001 --eval_iter 10 --save_iter 10 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 0 \
#     --desc 'decoder_no_in_ckpt' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_cache \
#     --code_cache_dir "runs/zerp123plus_batch/workspace_ablation/20240314-ablation4-fixed-encode-range-4gpus-resume-e30+180-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0_lpips1.0-lr0.0002-Plat100/code_dir" \
#     --resume "runs/zerp123plus_batch/workspace_ablation/20240320-ablation4_unet_fixed_encode_range-3gpus-resume_ep120-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0-lr4e-06-Plat100/eval_epoch_2000/model.safetensors"
#     # --resume "runs/zerp123plus_batch/workspace_ablation/20240321-ablation4_unet_fixed_encode_range-4gpus-resumeunet20240320_ep140-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0-lr2e-06-Plat100/eval_epoch_620/model.safetensors"
#     # --resume "runs/zerp123plus_batch/workspace_ablation/20240320-ablation4_unet_fixed_encode_range-3gpus-resume_ep120-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0-lr4e-06-Plat100/eval_epoch_140/model.safetensors"
#     # --resume "runs/zerp123plus_batch/workspace_ablation/20240316-ablation4_unet_fixed_encode_range-4gpus-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0-lr1e-05-Plat100/eval_epoch_120/model.safetensors"

# # [INFERENCE fixed VAE]
# # export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference.py big --workspace runs/zerp123plus_batch/workspace_debug \
#     --lr 2e-4 --num_epochs 10001 --eval_iter 5 --save_iter 5 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 0 \
#     --desc 'debug_encode_splatter' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_diffusion --vae_on_splatter_image 

# # [INFERENCE fixed VAE]
# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference_marigold.py big --workspace runs/marigold/workspace_debug \
#     --lr 2e-3 --num_epochs 10001 --eval_iter 5 --save_iter 5 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'debug_encode_splatter' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_diffusion --vae_on_splatter_image 


# [v3: optimzie latents]
# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference_marigold_v3_fake_init.py big --workspace runs/marigold/workspace_debug \
#     --lr 2e-3 --num_epochs 10001 --eval_iter 5 --save_iter 5 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'v3_fake_init_only_optimize_depth' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_diffusion --vae_on_splatter_image --group_scale --render_input_views \
#     --output_size 128 --input_size 128 \
#     --splatter_to_encode "runs/marigold/workspace_train/20240422-234007-v2_fake_init-vae_on_splatter_image-codes_from_diffusion-loss_render1.0_lpips1.0-lr0.002-Plat/zero123plus/outputs_v3_inference_my_decoder/0_02690cf0a24e49499b44fb4cb4dd3e68/100"

# \
    # --splatter_to_encode "runs/marigold/workspace_train/20240422-234007-v2_fake_init-vae_on_splatter_image-codes_from_diffusion-loss_render1.0_lpips1.0-lr0.002-Plat/zero123plus/outputs_v3_inference_my_decoder/0_02690cf0a24e49499b44fb4cb4dd3e68/100"

# # [v4: optimzie 3 channels splatters]
# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference_marigold_v4_fake_init_optimize_splatter.py big --workspace runs/marigold/workspace_debug \
#     --lr 2e-3 --num_epochs 10001 --eval_iter 5 --save_iter 5 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'v3_fake_init_only_optimize_depth' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_diffusion --vae_on_splatter_image --group_scale --render_input_views \
#     --output_size 128 --input_size 128 \
#     --optimization_objective "splatter_images" 


# # [v5: optimzie original channels splatters]
# # export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference_marigold_v5_fake_init_optimize_splatter.py big --workspace runs/marigold/workspace_debug \
#     --lr 2e-3 --num_epochs 1001 --eval_iter 5 --save_iter 5 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'v3_fake_init_scene1andAfter' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_diffusion --vae_on_splatter_image --group_scale --render_input_views \
#     --output_size 128 --input_size 128 \
#     --optimization_objective "splatter_images" 
    
# # export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference_marigold_v5_fake_init_optimize_splatter.py big --workspace runs/marigold/workspace_optimize \
#     --lr 2e-3 --num_epochs 301 --eval_iter 5 --save_iter 5 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --num_input_views 6 --num_views 55 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'v5_LGM_init_scene0andAfter_reg_encoder_input_every_iter_no_clip' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_diffusion --vae_on_splatter_image --group_scale --render_input_views \
#     --output_size 128 --input_size 128 \
#     --optimization_objective "splatter_images" --attr_group_mode "v5" \
#     --rendering_loss_on_splatter_to_encode --loss_weights_decoded_splatter 1.5 \
#     --scene_start_index 858  --scene_end_index 1000 \
#     --resume_workspace "runs/marigold/workspace_optimize/20240426-135026-v5_LGM_init_render320_scene_800_1000_reg_encoder_input_every_iter_no_clip-vae_on_splatter_image-codes_from_diffusion-loss_render1.0_lpips1.0-lr0.006-Plat"

# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference_marigold_v5_fake_init_optimize_splatter.py big --workspace runs/marigold/workspace_optimize \
#     --lr 2e-3 --num_epochs 301 --eval_iter 5 --save_iter 5 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --num_input_views 6 --num_views 55 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'v5_LGM_init_scene0andAfter_reg_encoder_input_every_iter_no_clip' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_diffusion --vae_on_splatter_image --group_scale --render_input_views \
#     --output_size 128 --input_size 128 \
#     --optimization_objective "splatter_images" --attr_group_mode "v5" \
#     --rendering_loss_on_splatter_to_encode --loss_weights_decoded_splatter 1.5 \
#     --scene_start_index 440  --scene_end_index 600 \
#     --resume_workspace "runs/marigold/workspace_optimize/20240426-140725-v5_LGM_init_render320_scene_400_600_reg_encoder_input_every_iter_no_clip-vae_on_splatter_image-codes_from_diffusion-loss_render1.0_lpips1.0-lr0.006-Plat"


# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference_marigold_v5_fake_init_optimize_splatter.py big --workspace runs/marigold/workspace_optimize \
#     --lr 2e-3 --num_epochs 301 --eval_iter 5 --save_iter 5 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --num_input_views 6 --num_views 55 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'v5_LGM_init_scene0andAfter_reg_encoder_input_every_iter_no_clip' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_diffusion --vae_on_splatter_image --group_scale --render_input_views \
#     --output_size 128 --input_size 128 \
#     --optimization_objective "splatter_images" --attr_group_mode "v5" \
#     --rendering_loss_on_splatter_to_encode --loss_weights_decoded_splatter 1.5 \
#     --scene_start_index 640  --scene_end_index 800 \
#     --resume_workspace "runs/marigold/workspace_optimize/20240426-140907-v5_LGM_init_render320_scene_600_800_reg_encoder_input_every_iter_no_clip-vae_on_splatter_image-codes_from_diffusion-loss_render1.0_lpips1.0-lr0.006-Plat"


# # # [APR 25 rendering at size 320. Frozen: for optimizing splatter images from LGM init.]
# # export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference_marigold_v5_fake_init_optimize_splatter.py big --workspace runs/marigold/workspace_debug \
#     --lr 6e-3 --num_epochs 301 --eval_iter 5 --save_iter 5 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --num_input_views 6 --num_views 55 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'v5_LGM_FIXROT_optimized_inference' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT_FIX_ROT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_diffusion --vae_on_splatter_image --group_scale --render_input_views \
#     --output_size 320 --input_size 320 \
#     --optimization_objective "splatter_images" --attr_group_mode "v5" \
#     --rendering_loss_on_splatter_to_encode \
#     --scene_start_index 0  --scene_end_index 200 --verbose 

    # --desc 'v5_LGM_init_render320_scene_0_100_reg_encoder_input_every_iter_no_clip'
    # --scene_start_index 0  --scene_end_index 100

# # # [APR 25 Inference]
# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference_marigold_v7_fake_init_optimize_splatter_inference_finetuned.py big --workspace runs/marigold/workspace_debug \
#     --lr 6e-3 --num_epochs 301 --eval_iter 5 --save_iter 5 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --num_input_views 6 --num_views 55 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'rgb-attr' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_diffusion --vae_on_splatter_image --group_scale --render_input_views \
#     --output_size 128 --input_size 128 \
#     --optimization_objective "splatter_images" --attr_group_mode "v5" \
#     --scene_start_index 0  --scene_end_index 5 \
#     --load_suffix "to_encode" --load_ext "png" --load_iter 1000 \
#     --custom_pipeline "./zero123plus/pipeline_v5.py" \
#     --resume "/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold_unet/workspace_CD_train/20240504-marigold-unet-single-attr-rgbs-train-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_lpips1.0-lr0.0001-Plat5/model.safetensors" 
#     # --splatter_to_encode "/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold/workspace_test/reg-both-encode-every-iter-20240425-003322-v5_LGM_init_scene0andAfter_reg_encoder_input_every_iter_no_clip-vae_on_splatter_image-codes_from_diffusion-loss_render1.0_lpips1.0-lr0.002-Plat/zero123plus/outputs_v3_inference_my_decoder"
#     # --resume "/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold_unet/workspace_CD_ovft/20240503-235752-marigold-unet-v-pred-cd-spatial-cat-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_lpips1.0-lr0.0001-Plat5/eval_epoch_3600/model.safetensors" \
#     # --rendering_loss_on_splatter_to_encode \
#     # --splatter_to_encode "/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold/workspace_test/reg-both-encode-save-iters-decode-20240424-184835-v5_LGM_init_scene1andAfter_to_decoded_png_inference1000-vae_on_splatter_image-codes_from_diffusion-loss_render1.0_lpips1.0-lr0.002-Plat/zero123plus/outputs_v3_inference_my_decoder"
#     # --splatter_to_encode "/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold/workspace_test/20240424-184835-v5_LGM_init_scene1andAfter_to_decoded_png_inference1000-vae_on_splatter_image-codes_from_diffusion-loss_render1.0_lpips1.0-lr0.002-Plat/zero123plus/outputs_v3_inference_my_decoder/1_02b0456362f9442da46d39fb34b3ee5b/800"
#     # --splatter_to_encode "/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold/workspace_optimize/20240425-232930-v5_LGM_init_render320_scene_0_200_reg_encoder_input_every_iter_no_clip-vae_on_splatter_image-codes_from_diffusion-loss_render1.0_lpips1.0-lr0.006-Plat/zero123plus/outputs_v3_inference_my_decoder"

# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference_marigold_v7_fake_init_optimize_splatter_inference_finetuned.py big --workspace runs/marigold_unet/workspace_CD_inference \
#     --lr 6e-3 --num_epochs 301 --eval_iter 5 --save_iter 5 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --num_input_views 6 --num_views 55 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'spatial-cat-train-epoch130-30steps' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_diffusion --vae_on_splatter_image --group_scale --render_input_views \
#     --output_size 128 --input_size 128 \
#     --optimization_objective "splatter_images" --attr_group_mode "v5" \
#     --scene_start_index 5  --scene_end_index 100 \
#     --load_suffix "to_encode" --load_ext "png" --load_iter 1000 \
#     --custom_pipeline "./zero123plus/pipeline_v5.py" \
#     --cd_spatial_concat \
#     --resume "/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold_unet/workspace_CD_train/20240505-4-gpus-marigold-unet-v-pred-cd-spatial-cat-all-data-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_lpips1.0-lr0.0001-Plat5/eval_epoch_100/model.safetensors"
  

# # export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference_marigold_v7_fake_init_optimize_splatter_inference_finetuned.py big --workspace runs/marigold_unet/workspace_CD_inference \
#     --lr 6e-3 --num_epochs 301 --eval_iter 5 --save_iter 5 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --num_input_views 6 --num_views 55 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'weight_t-pos-embed-sqeuence-cat-train-epoch200-t=50' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_diffusion --vae_on_splatter_image --group_scale --render_input_views \
#     --output_size 128 --input_size 128 \
#     --optimization_objective "splatter_images" --attr_group_mode "v5" \
#     --scene_start_index 5  --scene_end_index 100 \
#     --load_suffix "to_encode" --load_ext "png" --load_iter 1000 \
#     --custom_pipeline "./zero123plus/pipeline_v7_seq.py" \
#     --resume "/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold_unet/workspace_CD_train/20240518-8gpus-marigold-unet-v7_seq-POS-EMBED-resume160-rendering_loss_weight_alpha^2-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_lpips1.0-lr0.0001-Plat5/eval_epoch_200/model.safetensors"
#     # --resume "runs/marigold_unet/workspace_CD_train/20240519-LARGE-LR-Attn-only_attn-rendering_w_t-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_lpips1.0-lr0.0001-Plat5/eval_epoch_200/model.safetensors"
#     # --resume "/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold_unet/workspace_CD_train/20240517-8gpus-marigold-unet-v7_seq-POS-EMBED-only-train-attn-v-pred-white-bg-unified-t-resume160-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_lpips1.0-lr0.0001-Plat5/eval_epoch_210/model.safetensors"
#     # --resume "/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold_unet/workspace_CD_train/20240516-8gpus-marigold-unet-v7_seq-POS-EMBED-only-train-attn-v-pred-white-bg-unified-t-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_lpips1.0-lr0.0001-Plat5/eval_epoch_160/model.safetensors"
#     # --resume "/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold_unet/workspace_CD_train/20240510-205912-marigold-unet-only-train-attn-unified-t-save-epoch0-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_lpips1.0-lr0.0001-Plat5/eval_epoch_0/model.safetensors"
#     # --resume "/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold_unet/workspace_CD_inference/wrong-order-2steps-20240510-183300-sqeuence-cat-train-epoch100-30steps-vae_on_splatter_image-codes_from_diffusion-loss_render1.0_lpips1.0-lr0.006-Plat/zero123plus/outputs_v3_inference_my_decoder/5_02690cf0a24e49499b44fb4cb4dd3e68/0/opacity_decoded.png"
#     # --resume "/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold_unet/workspace_CD_train/20240515-marigold-unet-v7_seq-only-train-attn-v-pred-white-bg-unified-t-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_lpips1.0-lr0.0001-Plat5/eval_epoch_100/model.safetensors"
  
accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference_marigold_v7_fake_init_optimize_splatter_inference_finetuned.py big --workspace runs/marigold_unet/workspace_CD_inference \
    --lr 6e-3 --num_epochs 301 --eval_iter 5 --save_iter 5 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
    --prob_cam_jitter 0 --num_input_views 6 --num_views 55 \
    --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
    --desc 'SCHEDULE-LR-dev_cleanCodeForZero1toG-weight_t-pos-embed-sqeuence-cat-train-epoch280-30steps' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
    --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
    --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
    --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
    --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
    --decoder_upblocks_interpolate_mode "last_layer" \
    --codes_from_diffusion --vae_on_splatter_image --group_scale --render_input_views \
    --output_size 128 --input_size 128 \
    --optimization_objective "splatter_images" --attr_group_mode "v5" \
    --scene_start_index 5  --scene_end_index 100 \
    --load_suffix "to_encode" --load_ext "png" --load_iter 1000 \
    --custom_pipeline "./zero123plus/pipeline_v7_seq.py" \
    --resume "runs/marigold_unet/workspace_CD_train/20240520-SCHEDULE-LR-Attn-only_attn-rendering_w_t-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_lpips1.0-lr0.0001-Plat5/eval_epoch_280/model.safetensors"
    # --resume "runs/marigold_unet/workspace_CD_train/20240519-LARGE-LR-Attn-only_attn-rendering_w_t-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_lpips1.0-lr0.0001-Plat5/eval_epoch_300/model.safetensors"
    # --resume "/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold_unet/workspace_CD_train/20240518-8gpus-marigold-unet-v7_seq-POS-EMBED-resume160-rendering_loss_weight_alpha^2-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_lpips1.0-lr0.0001-Plat5/eval_epoch_200/model.safetensors"


# # [MAY 07]
# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference_marigold_v7_fake_init_optimize_splatter_inference_finetuned.py big --workspace runs/marigold_unet/workspace_CD_inference \
#     --lr 6e-3 --num_epochs 301 --eval_iter 5 --save_iter 5 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --num_input_views 6 --num_views 55 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'sequence-cat-train-epoch490-debug_t=50' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_diffusion --vae_on_splatter_image --group_scale --render_input_views \
#     --output_size 128 --input_size 128 \
#     --optimization_objective "splatter_images" --attr_group_mode "v5" \
#     --scene_start_index 0  --scene_end_index 100 \
#     --load_suffix "to_encode" --load_ext "png" --load_iter 1000 \
#     --custom_pipeline "./zero123plus/pipeline_v7_seq.py" \
#     --resume "/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold_unet/workspace_CD_train/20240514-marigold-unet-v7_seq-only-train-attn-v-pred-white-bg-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_lpips1.0-lr0.0001-Plat5/eval_epoch_490/model.safetensors"


# # [MAY 10]
# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference_marigold_v7_fake_init_optimize_splatter_inference_finetuned.py big --workspace runs/marigold_unet/workspace_CD_inference \
#     --lr 6e-3 --num_epochs 301 --eval_iter 5 --save_iter 5 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --num_input_views 6 --num_views 55 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'sequence-cat-train-epoch490-scheduler_onestep_t=50' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_diffusion --vae_on_splatter_image --group_scale --render_input_views \
#     --output_size 128 --input_size 128 \
#     --optimization_objective "splatter_images" --attr_group_mode "v5" \
#     --scene_start_index 0  --scene_end_index 100 \
#     --load_suffix "to_encode" --load_ext "png" --load_iter 1000 \
#     --custom_pipeline "./zero123plus/pipeline_v7_seq.py" \
#     --resume "/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold_unet/workspace_CD_train/20240514-marigold-unet-v7_seq-only-train-attn-v-pred-white-bg-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_lpips1.0-lr0.0001-Plat5/eval_epoch_490/model.safetensors"


# # # [resize optimzied splatter from 128 to 320]
# # export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference_marigold_v6_inference_resize_splatter.py big --workspace runs/marigold/workspace_debug \
#     --lr 2e-3 --num_epochs 1001 --eval_iter 5 --save_iter 5 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --num_input_views 6 --num_views 55 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'v6_directly_resize_optimized128_to320_render320' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_diffusion --vae_on_splatter_image --group_scale --render_input_views \
#     --output_size 320 --input_size 320 --splatter_size 320 \
#     --optimization_objective "splatter_images" --attr_group_mode "v5" \
#     --rendering_loss_on_splatter_to_encode \
#     --scene_start_index 0  --scene_end_index 4 \
#     --load_suffix "to_encode" --load_ext "png" --load_iter 300 \
#     --splatter_to_encode "runs/marigold/workspace_optimize/20240425-232930-v5_LGM_init_render320_scene_0_200_reg_encoder_input_every_iter_no_clip-vae_on_splatter_image-codes_from_diffusion-loss_render1.0_lpips1.0-lr0.006-Plat/zero123plus/outputs_v3_inference_my_decoder"
# #     # --splatter_to_encode "/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold/workspace_test/reg-both-encode-save-iters-decode-20240424-184835-v5_LGM_init_scene1andAfter_to_decoded_png_inference1000-vae_on_splatter_image-codes_from_diffusion-loss_render1.0_lpips1.0-lr0.002-Plat/zero123plus/outputs_v3_inference_my_decoder/1_02b0456362f9442da46d39fb34b3ee5b/1000"
# #     # --splatter_to_encode "/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold/workspace_debug/20240425-185946-v5_LGM_init_scene0andAfter_reg_encoder_input_every_iter_no_clip-vae_on_splatter_image-codes_from_diffusion-loss_render1.0_lpips1.0-lr0.002-Plat/zero123plus/outputs_v3_inference_my_decoder/3_032b9ee505844b46b3b1436de5fef3bb/270_success" \
    

# # [KEY NOTES: 
# # 1. load all the views
# # 2. also load the mask]
# # 3. add color: not for now, hard to acquire the mask for srn cars
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference_marigold_v5_fake_init_optimize_splatter.py big --workspace runs/marigold/workspace_debug \
#     --lr 2e-3 --num_epochs 1001 --eval_iter 5 --save_iter 5 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --num_input_views 6 --num_views 50 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'v5_LGM_init_srn_cars_reg_encoder_input_every_iter_no_clip' --data_path_rendering ${DATA_DIR_BATCH_RENDERING_SRN} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT_SRN_HQ} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_diffusion --vae_on_splatter_image --group_scale --render_input_views \
#     --output_size 128 --input_size 128 \
#     --optimization_objective "splatter_images" --attr_group_mode "v5" \
#     --rendering_loss_on_splatter_to_encode --data_mode srn_cars
    
    # \
    # --load_suffix "to_encode" --load_ext "png" --load_iter 1000 
    
    # --clip_image_to_encode 
    
    # \
    # 
    # \
    # --splatter_to_encode "/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold/workspace_test/reg-both-encode-save-iters-decode-20240424-184835-v5_LGM_init_scene1andAfter_to_decoded_png_inference1000-vae_on_splatter_image-codes_from_diffusion-loss_render1.0_lpips1.0-lr0.002-Plat/zero123plus/outputs_v3_inference_my_decoder"
    # --desc 'v5_LGM_init_scene0andAfter_inference_all_scenes'

    # --splatter_to_encode "/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold/workspace_test/reg-both-encode-save-iters-decode-20240424-184835-v5_LGM_init_scene1andAfter_to_decoded_png_inference1000-vae_on_splatter_image-codes_from_diffusion-loss_render1.0_lpips1.0-lr0.002-Plat/zero123plus/outputs_v3_inference_my_decoder/3_032b9ee505844b46b3b1436de5fef3bb/1000"

    # --splatter_to_encode "/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold/workspace_test/20240424-184835-v5_LGM_init_scene1andAfter_to_decoded_png_inference1000-vae_on_splatter_image-codes_from_diffusion-loss_render1.0_lpips1.0-lr0.002-Plat/zero123plus/outputs_v3_inference_my_decoder/1_02b0456362f9442da46d39fb34b3ee5b/800"
    
    # \
    # --load_suffix "decoded" --load_ext "png" \
    # --load_suffix "to_encode" --load_ext "png" \
    # --load_suffix "to_encode" --load_ext "pt" \
    # --splatter_to_encode "/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold/workspace_debug/20240424-175426-v5_LGM_init_scene1andAfter_clip_to_encode-vae_on_splatter_image-codes_from_diffusion-loss_render1.0_lpips1.0-lr0.002-Plat/zero123plus/outputs_v3_inference_my_decoder/1_02b0456362f9442da46d39fb34b3ee5b/1000"

    # \
    # --splatter_to_encode "/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold/workspace_debug/20240424-131010-v5_LGM_init_scene1andAfter_check-vae_on_splatter_image-codes_from_diffusion-loss_render1.0_lpips1.0-lr0.002-Plat/zero123plus/outputs_v3_inference_my_decoder/1_02b0456362f9442da46d39fb34b3ee5b/100"

    # --splatter_to_encode "/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold/workspace_train/20240423-193714-v5_LGM_init_scene1andAfter-vae_on_splatter_image-codes_from_diffusion-loss_render1.0_lpips1.0-lr0.002-Plat/zero123plus/outputs_v3_inference_my_decoder/1_02b0456362f9442da46d39fb34b3ee5b/1000"
    
    # --splatter_to_encode "/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold/workspace_debug/20240423-193714-v5_LGM_init_scene1andAfter-vae_on_splatter_image-codes_from_diffusion-loss_render1.0_lpips1.0-lr0.002-Plat"
    # --splatter_to_encode "LGM_init"
    # --desc 'v3_fake_init_only_optimize_depth'
    

    # --resume "runs/zerp123plus_batch/workspace_ablation/20240322-175501-unet-w-rendering-loss-resume_unet_20240320-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-numV6-loss_render1.0_splatter1.0_lpips1.0-lr1e-05-Plat5/model.safetensors"

    # \

    # --resume 'runs/zerp123plus_batch/workspace_ablation/20240315-ablation_2_fixed_encode_range_2gpus_lpips-resume20240312-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0_lpips1.0-lr0.0002-Plat100/eval_epoch_60/model.safetensors'
    # --resume "runs/zerp123plus_batch/workspace_debug/20240327-165142-ablation_2_fixed_encode_range_overfit_scene2-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0_lpips1.0-lr0.0001-Plat100/model.safetensors"
    # # "runs/zerp123plus_batch/workspace_ablation/20240328-ablation_2dot1_fixed_encode_range_4gpus-mix_diffusion_interval_10-resume20240315-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-numV26-loss_render1.0_splatter1.0_lpips1.0-lr0.0001-Plat100/model.safetensors"
    # 'runs/zerp123plus_batch/workspace_ablation/20240312-ablation_2_fixed_encode_range_2gpus_lpips-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0_lpips1.0-lr0.0002-Plat100/eval_epoch_40/model.safetensors'
    # "runs/zerp123plus_batch/workspace_ablation/20240326-134858-ablation_2_fixed_encode_range_overfit_scene2-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0_lpips1.0-lr0.0001-Plat100/model.safetensors"
    # --resume "runs/zerp123plus_batch/workspace_ablation/20240322-175501-unet-w-rendering-loss-resume_unet_20240320-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-numV6-loss_render1.0_splatter1.0_lpips1.0-lr1e-05-Plat5/model.safetensors"
    # --lipschitz_coefficient 0.6 --lipschitz_mode gaussian_noise 
    # \
    # --code_cache_dir "runs/zerp123plus_batch/workspace_ablation/20240314-ablation4-fixed-encode-range-4gpus-resume-e30+180-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0_lpips1.0-lr0.0002-Plat100/code_dir" \
    # --resume "runs/zerp123plus_batch/workspace_debug/20240322-175501-unet-w-rendering-loss-resume_unet_20240320-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-numV6-loss_render1.0_splatter1.0_lpips1.0-lr1e-05-Plat5/eval_epoch_110/model.safetensors"
    # # --resume "runs/zerp123plus_batch/workspace_ablation/20240322-195548-unet-w-rendering-loss-resume_unet_w_rendering20240322-022336-epo30-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-numV6-loss_render1.0_splatter1.0_lpips1.0-lr1e-05-Plat5/eval_epoch_90/model.safetensors"
    # # --resume "runs/zerp123plus_batch/workspace_ablation/20240322-022336-unet-w-rendering-loss-resume_unet_20240320-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-numV6-loss_render1.0_splatter1.0_lpips1.0-lr1e-05-Plat5/eval_epoch_30/model.safetensors"
    # # --resume "runs/zerp123plus_batch/workspace_lora/00002-lora-lora_rank128-gpus4-sp_guide_1-loss_render1.0-lr1e-05-Plat100/eval_epoch_140/model.safetensors"
