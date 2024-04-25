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
    
export CUDA_VISIBLE_DEVICES=1
accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference_marigold_v5_fake_init_optimize_splatter.py big --workspace runs/marigold/workspace_debug \
    --lr 2e-3 --num_epochs 1001 --eval_iter 5 --save_iter 5 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
    --prob_cam_jitter 0 --num_input_views 6 --num_views 55 \
    --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
    --desc 'v5_LGM_init_scene0andAfter_reg_encoder_input_every_iter_no_clip' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
    --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
    --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
    --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
    --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
    --decoder_upblocks_interpolate_mode "last_layer" \
    --codes_from_diffusion --vae_on_splatter_image --group_scale --render_input_views \
    --output_size 128 --input_size 128 \
    --optimization_objective "splatter_images" --attr_group_mode "v5" \
    --rendering_loss_on_splatter_to_encode \
    --scene_start_index 10  --scene_end_index 100


# # [try size 320]
# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference_marigold_v5_fake_init_optimize_splatter.py big --workspace runs/marigold/workspace_debug \
#     --lr 2e-3 --num_epochs 1001 --eval_iter 5 --save_iter 5 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --num_input_views 6 --num_views 55 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'v5_LGM_320_init_scene0andAfter_reg_encoder_input_every_iter_no_clip' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_diffusion --vae_on_splatter_image --group_scale --render_input_views \
#     --output_size 320 --input_size 320 \
#     --optimization_objective "splatter_images" --attr_group_mode "v5" \
#     --rendering_loss_on_splatter_to_encode \
#     --scene_start_index 0  --scene_end_index 4

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
