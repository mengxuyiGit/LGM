DATA_DIR_BATCH_RENDERING='/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999'
DATA_DIR_BATCH_SPLATTER_GT_ROOT='/home/xuyimeng/Repo/LGM/runs/splatter_gt'

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
    
# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference_marigold_v5_fake_init_optimize_splatter.py big --workspace runs/marigold/workspace_debug \
#     --lr 2e-3 --num_epochs 1001 --eval_iter 5 --save_iter 5 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
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
#     --scene_start_index 1  --scene_end_index 100


# # # [APR 25 rendering at size 320. Frozen: for optimizing splatter images from LGM init.]
# # export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference_marigold_v5_fake_init_optimize_splatter.py big --workspace runs/marigold/workspace_optimize \
#     --lr 6e-3 --num_epochs 301 --eval_iter 5 --save_iter 5 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --num_input_views 6 --num_views 55 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'v5_LGM_init_render320_scene_0_200_reg_encoder_input_every_iter_no_clip' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_diffusion --vae_on_splatter_image --group_scale --render_input_views \
#     --output_size 320 --input_size 320 \
#     --optimization_objective "splatter_images" --attr_group_mode "v5" \
#     --rendering_loss_on_splatter_to_encode \
#     --scene_start_index 0  --scene_end_index 200

    # --desc 'v5_LGM_init_render320_scene_0_100_reg_encoder_input_every_iter_no_clip'
    # --scene_start_index 0  --scene_end_index 100

# # [APR 25 rendering at size 320. Frozen: for optimizing splatter images from LGM init.]
# # export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference_marigold_v5_fake_init_optimize_splatter.py big --workspace runs/marigold/workspace_optimize \
#     --lr 6e-3 --num_epochs 301 --eval_iter 5 --save_iter 5 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --num_input_views 6 --num_views 55 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'v5_LGM_init_render320_scene_400_600_reg_encoder_input_every_iter_no_clip' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_diffusion --vae_on_splatter_image --group_scale --render_input_views \
#     --output_size 320 --input_size 320 \
#     --optimization_objective "splatter_images" --attr_group_mode "v5" \
#     --rendering_loss_on_splatter_to_encode \
#     --scene_start_index 400  --scene_end_index 600


# # export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference_marigold_v5_fake_init_optimize_splatter.py big --workspace runs/marigold_srn/workspace_optimize \
#     --lr 6e-3 --num_epochs 601 --eval_iter 5 --save_iter 5 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --num_input_views 6 --num_views 50 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'v5_LGM_init_render320_srn_0_500_reg_encode' --data_path_rendering ${DATA_DIR_BATCH_RENDERING_SRN} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT_SRN_HQ} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_diffusion --vae_on_splatter_image --group_scale --render_input_views \
#     --output_size 320 --input_size 320 \
#     --optimization_objective "splatter_images" --attr_group_mode "v5" \
#     --rendering_loss_on_splatter_to_encode \
#     --scene_start_index 28  --scene_end_index 500 --data_mode "srn_cars" 
#     # \
#     # --resume_workspace "runs/marigold_srn/workspace_optimize/20240427-191345-v5_LGM_init_render320_srn_1500_2000_reg_encode-vae_on_splatter_image-codes_from_diffusion-loss_render1.0_lpips1.0-lr0.01-Plat"

# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference_marigold_v5_fake_init_optimize_splatter.py big --workspace runs/marigold_srn/workspace_optimize \
#     --lr 6e-3 --num_epochs 601 --eval_iter 5 --save_iter 5 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --num_input_views 6 --num_views 50 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'v5_LGM_init_render320_srn_500_1000_reg_encode' --data_path_rendering ${DATA_DIR_BATCH_RENDERING_SRN} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT_SRN_HQ} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_diffusion --vae_on_splatter_image --group_scale --render_input_views \
#     --output_size 320 --input_size 320 \
#     --optimization_objective "splatter_images" --attr_group_mode "v5" \
#     --rendering_loss_on_splatter_to_encode \
#     --scene_start_index 526  --scene_end_index 1000 --data_mode "srn_cars"


# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference_marigold_v5_fake_init_optimize_splatter.py big --workspace runs/marigold_srn/workspace_optimize \
#     --lr 1e-2 --num_epochs 601 --eval_iter 5 --save_iter 5 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --num_input_views 6 --num_views 50 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'v5_LGM_init_render320_srn_1000_1500_reg_encode' --data_path_rendering ${DATA_DIR_BATCH_RENDERING_SRN} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT_SRN_HQ} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_diffusion --vae_on_splatter_image --group_scale --render_input_views \
#     --output_size 320 --input_size 320 \
#     --optimization_objective "splatter_images" --attr_group_mode "v5" \
#     --rendering_loss_on_splatter_to_encode \
#     --scene_start_index 1028  --scene_end_index 1500 --data_mode "srn_cars"


# # dl-a40 -1
# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference_marigold_v5_fake_init_optimize_splatter.py big --workspace runs/marigold_srn/workspace_optimize \
#     --lr 1e-2 --num_epochs 601 --eval_iter 5 --save_iter 5 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --num_input_views 6 --num_views 50 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'v5_LGM_init_render320_srn_1500_2000_reg_encode' --data_path_rendering ${DATA_DIR_BATCH_RENDERING_SRN} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT_SRN_HQ} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_diffusion --vae_on_splatter_image --group_scale --render_input_views \
#     --output_size 320 --input_size 320 \
#     --optimization_objective "splatter_images" --attr_group_mode "v5" \
#     --rendering_loss_on_splatter_to_encode \
#     --scene_start_index 1537  --scene_end_index 2000 --data_mode "srn_cars" 
#     # \
#     # --resume_workspace "runs/marigold_srn/workspace_optimize/20240427-191345-v5_LGM_init_render320_srn_1500_2000_reg_encode-vae_on_splatter_image-codes_from_diffusion-loss_render1.0_lpips1.0-lr0.01-Plat"

# # ll-40 -0
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference_marigold_v5_fake_init_optimize_splatter.py big --workspace runs/marigold_srn/workspace_optimize \
#     --lr 1e-2 --num_epochs 601 --eval_iter 5 --save_iter 5 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --num_input_views 6 --num_views 50 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'v5_LGM_init_render320_srn_300_500_reg_encode' --data_path_rendering ${DATA_DIR_BATCH_RENDERING_SRN} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT_SRN_HQ} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_diffusion --vae_on_splatter_image --group_scale --render_input_views \
#     --output_size 320 --input_size 320 \
#     --optimization_objective "splatter_images" --attr_group_mode "v5" \
#     --rendering_loss_on_splatter_to_encode \
#     --scene_start_index 300  --scene_end_index 500 --data_mode "srn_cars"

# # # ll-40 -1
# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference_marigold_v5_fake_init_optimize_splatter.py big --workspace runs/marigold_srn/workspace_optimize \
#     --lr 1e-2 --num_epochs 601 --eval_iter 5 --save_iter 5 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --num_input_views 6 --num_views 50 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'v5_LGM_init_render320_srn_800_1000_reg_encode' --data_path_rendering ${DATA_DIR_BATCH_RENDERING_SRN} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT_SRN_HQ} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_diffusion --vae_on_splatter_image --group_scale --render_input_views \
#     --output_size 320 --input_size 320 \
#     --optimization_objective "splatter_images" --attr_group_mode "v5" \
#     --rendering_loss_on_splatter_to_encode \
#     --scene_start_index 800  --scene_end_index 1000 --data_mode "srn_cars"


# # export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference_marigold_v5_fake_init_optimize_splatter.py big --workspace runs/marigold_srn/workspace_optimize \
#     --lr 1e-2 --num_epochs 601 --eval_iter 5 --save_iter 5 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --num_input_views 6 --num_views 50 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'v5_LGM_init_render320_srn_1647_2000_reg_encode' --data_path_rendering ${DATA_DIR_BATCH_RENDERING_SRN} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT_SRN_HQ} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_diffusion --vae_on_splatter_image --group_scale --render_input_views \
#     --output_size 320 --input_size 320 \
#     --optimization_objective "splatter_images" --attr_group_mode "v5" \
#     --rendering_loss_on_splatter_to_encode \
#     --scene_start_index 1647  --scene_end_index 2000 --data_mode "srn_cars" 

# [May 1: debug cam]
accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference_marigold_v5_fake_init_optimize_splatter.py big --workspace runs/marigold_srn/workspace_debug \
    --lr 1e-2 --num_epochs 1 --eval_iter 5 --save_iter 5 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
    --prob_cam_jitter 0 --num_input_views 6 --num_views 50 \
    --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
    --desc 'check_srn_6view_cam' --data_path_rendering ${DATA_DIR_BATCH_RENDERING_SRN} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT_SRN_HQ} \
    --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
    --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
    --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
    --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
    --decoder_upblocks_interpolate_mode "last_layer" \
    --codes_from_diffusion --vae_on_splatter_image --group_scale --render_input_views \
    --output_size 320 --input_size 320 \
    --optimization_objective "splatter_images" --attr_group_mode "v5" \
    --rendering_loss_on_splatter_to_encode \
    --scene_start_index 1647  --scene_end_index 2000 --data_mode "srn_cars" 