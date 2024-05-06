DATA_DIR_DESK='/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/0d83a6b0d3dc4a3b8544fff507c04d86'
DATA_DIR_PINK_IRONMAN='/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/1dbcffe2f80b4d3ca50ff6406ab81f84'
DATA_DIR_HYDRANT='/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/0a9b36d36e904aee8b51e978a7c0acfd'
DATA_DIR_LAMP='/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/0c58250e3a7242e9bf21b114f2c8dce6'

DATA_DIR_BATCH_RENDERING='/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999'
DATA_DIR_BATCH_SPLATTER_GT_ROOT='/home/xuyimeng/Repo/LGM/runs/splatter_gt'
# DATA_DIR_BATCH_SPLATTER_GT='/home/xuyimeng/Repo/LGM/data/splatter_gt_batch/subset_10-250/9000-9999'
# DATA_DIR_BATCH_SPLATTER_GT_10_250='/home/xuyimeng/Repo/LGM/runs/LGM_optimize_splatter/workspace_debug_batch_subset_es/00000-batch-es10-Plat-patience_2-factor_0.5-eval_5-adamW-subset_10_250_splat128-inV6-lossV20-lr0.003/9000-9999'
# DATA_DIR_BATCH_SPLATTER_GT_750_END='/home/xuyimeng/Repo/LGM/runs/LGM_optimize_splatter/workspace_debug_batch_subset_es/00005-batch-es10-Plat-patience_2-factor_0.5-eval_5-adamW-subset_750_-1_splat128-inV6-lossV20-lr0.003/9000-9999'
# DATA_DIR_BATCH_SPLATTER_GT_NO_RELATIVE_CAM="/home/xuyimeng/Repo/LGM/runs/LGM_optimize_splatter/workspace_debug_batch_subset_es/00010-batch-es10-Plat-patience_2-factor_0.5-eval_1-adamW-subset_1_5_splat128-inV6-lossV20-lr0.003-NO-REL-CAM/9000-9999"
# DATA_DIR_BATCH_SPLATTER_GT_USE_RELATIVE_CAM="/home/xuyimeng/Repo/LGM/runs/LGM_optimize_splatter/workspace_debug_batch_subset_es/00011-batch-es10-Plat-patience_2-factor_0.5-eval_1-adamW-subset_1_5_splat128-inV6-lossV20-lr0.003/9000-9999"
# DATA_DIR_BATCH_SPLATTER_GT_USE_RELATIVE_CAM_1_INPUT_V="/home/xuyimeng/Repo/LGM/runs/LGM_optimize_splatter/workspace_debug_batch_subset_es/00015-batch-es10-Plat-patience_2-factor_0.5-eval_1-adamW-subset_1_5_splat128-inV1-lossV20-lr0.003/9000-9999"
# DATA_DIR_BATCH_SPLATTER_GT_USE_RELATIVE_CAM_1_INPUT_V_START_V3="/home/xuyimeng/Repo/LGM/runs/LGM_optimize_splatter/workspace_debug_batch_subset_es/00016-batch-es10-Plat-patience_2-factor_0.5-eval_1-adamW-subset_1_5_splat128-inV1-lossV20-lr0.003/9000-9999"

# debug training: fix pretrained 
#  accelerate launch --config_file acc_configs/gpu1.yaml main_cw.py big \
#     --workspace workspace_debug_pretrained_tf \
#     --resume pretrained/model_fp16.safetensors --num_epochs 100 --fix_pretrained \
#     --lr 0.0001 --num_views 6 --desc 'cw_dataloader' --eval_iter 1

# accelerate launch --config_file acc_configs/gpu1.yaml main_pretrained.py big \
#     --workspace runs/workspace_debug_pretrained_tf \
#     --resume pretrained/model_fp16.safetensors --num_epochs 10001 --fix_pretrained \
#     --lr 0.0006 --num_input_views 6 --num_views 20 --desc 'desk' --eval_iter 100 \
#     --prob_cam_jitter 0 --data_path '/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/0d83a6b0d3dc4a3b8544fff507c04d86'

# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus.py big \
#     --workspace runs/zerp123plus_overfit/workspace_train \
#     --num_epochs 100 \
#     --lr 0.001 --num_input_views 6 --num_views 20 --desc 'debug' --eval_iter 1 \
#     --prob_cam_jitter 0 --data_path ${DATA_DIR_HYDRANT}

# ## overfit hygrant using zero123++, only L2 loss with splatter GT
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v3.py big --workspace runs/zerp123plus_overfit/workspace_overfit \
#     --num_epochs 10001 \
#     --lr 0.0001 --num_input_views 6 --num_views 20 --desc 'hydrant' --eval_iter 50 --save_iter 500\
#     --prob_cam_jitter 0 --data_path ${DATA_DIR_HYDRANT} --input_size 128 --lambda_splatter 1 --lambda_rendering 0 --lambda_lpips 0 \
#     --log_gs_loss_mse_dict --attr_use_logrithm_loss 'scale' 'opacity'

# ## [DEBUG] overfit hygrant using zero123++
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v3.py big --workspace runs/zerp123plus_overfit/workspace_debug \
#     --num_epochs 10001 \
#     --lr 0.0001 --num_input_views 6 --num_views 20 --desc 'hydrant' --eval_iter 5 --save_iter 1000 \
#     --prob_cam_jitter 0 --data_path ${DATA_DIR_HYDRANT} --input_size 128 --lambda_splatter 1 --lambda_rendering 0 --lambda_lpips 0 \
#     --perturb_rot_scaling --render_gt_splatter --attr_use_logrithm_loss 'scale' 'opacity'

# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch.py big --workspace runs/zerp123plus_batch/workspace_train \
#     --lr 2e-4 --num_epochs 10001 --eval_iter 50 --save_iter 50 --lr_scheduler Plat --lr_scheduler_patience 5 \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 0 \
#     --desc 'compare-hydrant-no-code-with-code-from-encoder' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussian \
#     --overfit_one_scene

## [DEBUG] optimize code
## debug 1 gpu
# export CUDA_VISIBLE_DEVICES=3
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code.py big --workspace runs/zerp123plus_batch/workspace_train \
#     --lr 2e-4 --num_epochs 10001 --eval_iter 100 --save_iter 100 --lr_scheduler Plat --lr_scheduler_patience 5 \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 0 \
#     --desc 'decode_splatter_to_128_nearest' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 10 --save_train_pred -1 --overfit_one_scene --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer"
    
    # \ --use_activation_at_downsample --downsample_mode "ConvAvgPool" 
    # --resume runs/zerp123plus_batch/workspace_debug_codedir/00001-optimize-code-not-splatter-w-guidance-large-code-lr-sp_guide_10-v0_unfreeze_all-skip_predict_x0-loss_render1.0_splatter1.0-lr0.0002-Plat5/model.safetensors

#     # \--verbose_main 
    # --resume runs/zerp123plus_batch/workspace_train/20240226-012340-train-resume185634-920-aligned-sf-2gpu-v0_unfreeze_all-skip_predict_x0-loss_render1.0_splatter1.0-lr0.0002-Plat5/model.safetensors
#     # --resume runs/zerp123plus_batch/workspace_train/20240225-185634-train-920-aligned-sf-v0_unfreeze_all-skip_predict_x0-loss_render1.0_splatter1.0-lr0.0002-Plat5/model.safetensors

#  --model_type Zero123PlusGaussianPosOffset --splatter_cfg.depth_scale 1.0

## debug 2 gpus
# accelerate launch --main_process_port 29513 --config_file acc_configs/gpu2.yaml main_zero123plus_v4_batch_code.py big --workspace runs/zerp123plus_batch/workspace_debug \
#     --lr 2e-4 --num_epochs 10001 --eval_iter 1 --save_iter 1 --lr_scheduler Plat --lr_scheduler_patience 5 \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 0 \
#     --desc 'debug-2gpus-workspace' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode --codes_from_encoder --overfit_one_scene


#  --skip_training 
#     # --decoder_mode v1_fix_rgb_remove_unscale
#     # --resume runs/zerp123plus_batch/workspace_train/20240223-164633-train-1gpu-10-250-clamp-norm-norm_scale-skip_predict_x0-loss_render1.0_splatter1.0-lr0.002/model.safetensors
#     # runs/zerp123plus_batch/workspace_debug/20240222-054252-train-no-unet-skip_predict_x0-loss_splatter1.0-lr0.002-good/model.safetensors


# # --------------- batch training ------------------
# ### gpus=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch.py big --workspace runs/zerp123plus_batch/workspace_train \
#     --lr 2e-4 --num_epochs 10001 --eval_iter 50 --save_iter 50 --lr_scheduler Plat --lr_scheduler_patience 5 \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 0 \
#     --desc 'overfit-scene10_250-bs4-latent16' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_10_250} \
#     --set_random_seed --batch_size 4 --num_workers 4 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10
# #     # \
    # --resume runs/zerp123plus_batch/workspace_train/20240226-012340-train-resume185634-920-aligned-sf-2gpu-v0_unfreeze_all-skip_predict_x0-loss_render1.0_splatter1.0-lr0.0002-Plat5/model.safetensors
    
    # --resume runs/zerp123plus_batch/workspace_train/20240226-013642-train-resume185634-920-aligned-sf-v0_unfreeze_all-skip_predict_x0-loss_render1.0_splatter1.0-lr0.0002-Plat5/model.safetensors
    # runs/zerp123plus_batch/workspace_train/20240225-185634-train-920-aligned-sf-v0_unfreeze_all-skip_predict_x0-loss_render1.0_splatter1.0-lr0.0002-Plat5/model.safetensors # the first best

# # ### gpus=2
# accelerate launch --main_process_port 29511 --config_file acc_configs/gpu2.yaml main_zero123plus_v4_batch.py big --workspace runs/zerp123plus_batch/workspace_train \
#     --lr 2e-4 --num_epochs 10001 --eval_iter 10 --save_iter 10 --lr_scheduler Plat --lr_scheduler_patience 5 \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 0 \
#     --desc '2gpus-overfit-scene10_250-bs4-latent16' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_10_250} \
#     --set_random_seed --batch_size 8 --num_workers 4 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --resume runs/zerp123plus_batch/workspace_train/20240301-213611-overfit-scene10_250_all_val-bs4-latent16-2gpus-v0_unfreeze_all-skip_predict_x0-loss_render1.0-lr0.0002-Plat5/model.safetensors
    # \
    # --resume runs/zerp123plus_batch/workspace_train/20240226-012340-train-resume185634-920-aligned-sf-2gpu-v0_unfreeze_all-skip_predict_x0-loss_render1.0_splatter1.0-lr0.0002-Plat5/model.safetensors

#     --resume runs/zerp123plus_batch/workspace_train/20240225-185634-train-920-aligned-sf-v0_unfreeze_all-skip_predict_x0-loss_render1.0_splatter1.0-lr0.0002-Plat5/model.safetensors
 # --resume runs/zerp123plus_batch/workspace_train/20240224-221548-train-920-clamp-skip_predict_x0-loss_render1.0_splatter1.0-lr0.0002/model.safetensors

    # --resume runs/zerp123plus_batch/workspace_train/20240224-194635-train-1gpu-10-250-clamp-norm_scale-skip_predict_x0-loss_render1.0-lr0.0002/model.safetensors
    # runs/zerp123plus_batch/workspace_train/20240224-203051-train-1gpu-10-250-clamp-skip_predict_x0-loss_render1.0_splatter1.0-lr0.0002/model.safetensors
    # runs/zerp123plus_batch/workspace_debug/20240222-054252-train-no-unet-skip_predict_x0-loss_splatter1.0-lr0.002-good/model.safetensors
    # --resume runs/zerp123plus_batch/workspace_train/20240223-164633-train-1gpu-10-250-clamp-norm-norm_scale-skip_predict_x0-loss_render1.0_splatter1.0-lr0.002/model.safetensors

    # --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable 
    # --scale_act 'softplus' --normalize_scale_using_gt 'scale'

# ## [DEBUG: resume ckpt]
# accelerate launch --config_file acc_configs/gpu2.yaml main_zero123plus_v4_batch_code_inference.py big --workspace runs/zerp123plus_batch/workspace_debug \
#     --lr 2e-4 --num_epochs 10001 --eval_iter 1 --save_iter 1 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 0 \
#     --desc 'resume-2gpus' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --resume "runs/zerp123plus_batch/workspace_train/20240308-4gpus_decode_splatter_to_128_nearest-sp_guide_10-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0-lr0.0002-Plat5/model.safetensors" 

    # --resume 'runs/zerp123plus_batch/workspace_train/20240303-075808-decode_splatter_to_128_last_layer_nearest-sp_guide_10-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0-lr0.0002-Plat5/model.safetensors' \
    # --overfit_one_scene

# ## [DEBUG: lora]
# ## 1gpus
# accelerate launch --config_file acc_configs/gpu4.yaml main_zero123plus_v4_batch_code_unet_lora.py big --workspace runs/zerp123plus_batch/workspace_lora \
#     --num_epochs 10001 --eval_iter 20 --save_iter 20 --lr_scheduler Plat \
#     --lr 1e-5 --min_lr_scheduled 1e-10 --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 0 \
#     --desc 'lora' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --lora_rank 128 \
#     --code_cache_dir "runs/zerp123plus_batch/workspace_ablation/20240314-ablation4-fixed-encode-range-4gpus-resume-e30+180-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0_lpips1.0-lr0.0002-Plat100/code_dir" \
#     --resume "runs/zerp123plus_batch/workspace_ablation/20240314-ablation4-fixed-encode-range-4gpus-resume-e30+180-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0_lpips1.0-lr0.0002-Plat100/eval_epoch_300/model.safetensors"

# # [TRAIN: ablation1: no splatter loss, no optimize latent code]
# accelerate launch --config_file acc_configs/gpu2.yaml main_zero123plus_v4_batch_code.py big --workspace runs/zerp123plus_batch/workspace_ablation \
#     --lr 2e-4 --num_epochs 10001 --eval_iter 20 --save_iter 20 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 0 \
#     --desc 'ablation1_2gpus_decode_splatter_to_128_nearest' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 2 --num_workers 2 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_encoder 
    # \
    # --skip_training --overfit_one_scene --save_iter 1 --eval_iter 2 # debug


# # # [TRAIN: ablation2: w/ splatter loss, no optimize latent code]
# accelerate launch --main_process_port 29511 --config_file acc_configs/gpu4.yaml main_zero123plus_v4_batch_code.py big --workspace runs/zerp123plus_batch/workspace_ablation \
#     --lr 1e-4 --num_epochs 20001 --eval_iter 20 --save_iter 20 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --prob_grid_distortion 0 --input_size 320 --num_input_views 6 --num_views 26 \
#     --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'ablation_2_fixed_encode_range_4gpus-resume20240315' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --resume 'runs/zerp123plus_batch/workspace_ablation/20240315-ablation_2_fixed_encode_range_2gpus_lpips-resume20240312-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0_lpips1.0-lr0.0002-Plat100/eval_epoch_60/model.safetensors'


# # # [TRAIN: ablation2: w/ splatter loss, no optimize latent code]
# ## single gpu, overfit one scene
# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code.py big --workspace runs/zerp123plus_batch/workspace_debug \
#     --lr 1e-4 --num_epochs 20001 --eval_iter 100 --save_iter 100 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --prob_grid_distortion 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'ablation_2_only_decoder_objaverse' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_encoder --output_size 128 --batch_size 2 --num_workers 2

# ## single gpu, overfit one scene, [SRN_cars]
# DATA_DIR_BATCH_RENDERING_SRN='/home/xuyimeng/Data/SRN/srn_cars/cars_train'
# DATA_DIR_BATCH_SPLATTER_GT_ROOT_SRN='/home/xuyimeng/Repo/LGM/runs/LGM_optimize_splatter/shapenet_fit_batch'
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code.py big --workspace runs/zerp123plus_batch/workspace_overfit \
#     --lr 1e-4 --num_epochs 20001 --eval_iter 100 --save_iter 100 \
#     --prob_cam_jitter 0 --prob_grid_distortion 0 --input_size 320 --num_input_views 6 --num_views 30 \
#     --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'ablation_2_only_decoder_srn_overfit_mask_loss' --data_path_rendering ${DATA_DIR_BATCH_RENDERING_SRN} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT_SRN} \
#     --set_random_seed --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --overfit_one_scene --data_mode "srn_cars" \
#     --codes_from_encoder --output_size 128 --batch_size 2 --num_workers 2 \
#     --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --render_input_views
    

# ## 2 gpus training [SRN_cars]
# DATA_DIR_BATCH_RENDERING_SRN='/home/xuyimeng/Data/SRN/srn_cars/cars_train'
# DATA_DIR_BATCH_SPLATTER_GT_ROOT_SRN='/home/xuyimeng/Repo/LGM/runs/LGM_optimize_splatter/shapenet_fit_batch'
# accelerate launch --main_process_port 29514 --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code.py big --workspace runs/decoder_only/workspace_ablation2 \
#     --lr 1e-4 --num_epochs 20001 --eval_iter 5 --save_iter 10 \
#     --prob_cam_jitter 0 --prob_grid_distortion 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'ablation_2_only_decoder_srn_cars_render_gt_splatter' --data_path_rendering ${DATA_DIR_BATCH_RENDERING_SRN} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT_SRN} \
#     --set_random_seed --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --data_mode "srn_cars" \
#     --codes_from_encoder --output_size 128 --batch_size 1 --num_workers 1 \
#     --render_input_views 
#     # --render_gt_splatter --skip_training

# ## single gpus training [SRN_cars]
# DATA_DIR_BATCH_RENDERING_SRN='/home/xuyimeng/Data/SRN/srn_cars/cars_train'
# DATA_DIR_BATCH_SPLATTER_GT_ROOT_SRN='/home/xuyimeng/Repo/LGM/runs/LGM_optimize_splatter/shapenet_fit_batch'
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code.py big --workspace runs/decoder_only/workspace_ablation2 \
#     --lr 1e-4 --num_epochs 20001 --eval_iter 5 --save_iter 10 \
#     --prob_cam_jitter 0 --prob_grid_distortion 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'ablation_2_only_decoder_srn_cars_1gpu_bsz2' --data_path_rendering ${DATA_DIR_BATCH_RENDERING_SRN} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT_SRN} \
#     --set_random_seed --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --data_mode "srn_cars" \
#     --codes_from_encoder --output_size 128 --batch_size 1 --num_workers 1 \
#     --render_input_views

    # --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 

# # # TODO: [TRAIN: ablation2.1: w/ splatter loss, no optimize latent code, and trained on mixed vae encoder output and diffusion output?]
# ## single gpu, overfit one scene
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code.py big --workspace runs/zerp123plus_batch/workspace_debug \
#     --lr 1e-4 --num_epochs 20001 --eval_iter 100 --save_iter 100 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --prob_grid_distortion 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'depth_offset' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder --overfit_one_scene \
#     --use_splatter_with_depth_offset

# 4gpus
# accelerate launch --main_process_port 29510 --config_file acc_configs/gpu4.yaml main_zero123plus_v4_batch_code.py big --workspace runs/zerp123plus_batch/workspace_ablation \
#     --lr 1e-4 --num_epochs 20001 --eval_iter 20 --save_iter 20 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --prob_grid_distortion 0 --input_size 320 --num_input_views 6 --num_views 26 \
#     --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'ablation_2dot1_fixed_encode_range_4gpus-mix_diffusion_interval_10-resume20240315' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder --mix_diffusion_interval 1 \
#     --resume 'runs/zerp123plus_batch/workspace_ablation/20240315-ablation_2_fixed_encode_range_2gpus_lpips-resume20240312-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0_lpips1.0-lr0.0002-Plat100/eval_epoch_60/model.safetensors'

# # [TRAIN: ablation4(ours): w/ splatter loss, w/ optimize latent code]
# # # 2gpus
# accelerate launch --main_process_port 29512 --config_file acc_configs/gpu4.yaml main_zero123plus_v4_batch_code.py big --workspace runs/zerp123plus_batch/workspace_ablation \
#     --lr 2e-4 --num_epochs 10001 --eval_iter 20 --save_iter 20 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'ablation4-fixed-encode-range-4gpus-resume-e30+180' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --resume 'runs/zerp123plus_batch/workspace_ablation/20240313-ablation4-fixed-encode-range-4gpus-resume-e30-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0-lr0.0002-Plat100/model.safetensors'

# # # [TRAIN diffusion unet - w/o rendering loss]
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_unet.py big --workspace runs/zerp123plus_batch/workspace_unet \
#     --lr 5e-5 --num_epochs 20001 --eval_iter 100 --save_iter 100 --lr_scheduler Plat \
#     --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 0 \
#     --desc '1-gpu-only-latent-loss-overfit_scene2_not_rand_unet' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCodeUnet \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --overfit_one_scene \
#     --resume "runs/zerp123plus_batch/workspace_train/20240315-4gpus_decode_splatter_to_128_nearest-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter0.5_lpips1.0-lr0.0002-Plat100/model.safetensors"

# # # ## 2gpus
# accelerate launch --main_process_port 29511 --config_file acc_configs/gpu2.yaml main_zero123plus_v4_batch_code_unet.py big --workspace runs/zerp123plus_batch/workspace_unet \
#     --lr 1e-6 --num_epochs 10001 --eval_iter 10 --save_iter 20 --lr_scheduler Plat \
#     --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 0 \
#     --desc '2-gpus-resume-unet20240317' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCodeUnet \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --resume_unet "runs/zerp123plus_batch/workspace_unet/20240317-4-gpus-resume-unet20240315-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0-lr1e-05-Plat100/model.safetensors" \
#     --min_lr_scheduled 1e-10 --resume "runs/zerp123plus_batch/workspace_train/20240315-4gpus_decode_splatter_to_128_nearest-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter0.5_lpips1.0-lr0.0002-Plat100/model.safetensors"
    
## 4gpus [CKPT]
# accelerate launch --config_file acc_configs/gpu4.yaml main_zero123plus_v4_batch_code_unet.py big --workspace runs/zerp123plus_batch/workspace_ablation \
#     --num_epochs 10001 --eval_iter 20 --save_iter 20 --lr_scheduler Plat \
#     --lr 2e-6 --min_lr_scheduled 1e-10 --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 0 \
#     --desc 'ablation4_unet_fixed_encode_range-4gpus-resumeunet20240320_ep140' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCodeUnet \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --resume_unet "runs/zerp123plus_batch/workspace_ablation/20240320-ablation4_unet_fixed_encode_range-3gpus-resume_ep120-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0-lr4e-06-Plat100/eval_epoch_140/model.safetensors" \
#     --resume "runs/zerp123plus_batch/workspace_ablation/20240314-ablation4-fixed-encode-range-4gpus-resume-e30+180-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0_lpips1.0-lr0.0002-Plat100/eval_epoch_300/model.safetensors"
    
    # --resume_unet "runs/zerp123plus_batch/workspace_unet/20240315-4-gpus-only-latent-loss-pretrained-unet-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0-lr5e-05-Plat100/model.safetensors" \
    # --resume_unet "runs/zerp123plus_batch/workspace_unet/20240309-4-gpus-only-latent-loss-resume20240308-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0-lr4e-05-Plat5/model.safetensors" \
    # --resume_unet "runs/zerp123plus_batch/workspace_unet/20240306-4-gpus-only-latent-loss-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0-lr0.0001-Plat100/model.safetensors" \

# # [TRAIN diffusion unet - use rendering loss]
# export CUDA_VISIBLE_DEVICES=0
accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_unet_rendering_loss.py big --workspace runs/zerp123plus_batch/workspace_debug \
    --lr 1e-5 --num_epochs 10001 --eval_iter 100 --save_iter 100 --lr_scheduler Plat \
    --lr_scheduler_patience 5 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
    --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 10 \
    --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
    --desc 'unet-w-rendering-loss-cosine_sched' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
    --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
    --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
    --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCodeUnet \
    --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
    --decoder_upblocks_interpolate_mode "last_layer" --overfit_one_scene --codes_from_encoder 
    
    # --scheduler_type "cosine" 
    # \
    # --resume "runs/zerp123plus_batch/workspace_ablation/20240314-ablation4-fixed-encode-range-4gpus-resume-e30+180-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0_lpips1.0-lr0.0002-Plat100/eval_epoch_300/model.safetensors"
    # --resume_unet "runs/zerp123plus_batch/workspace_ablation/20240322-022336-unet-w-rendering-loss-resume_unet_20240320-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-numV6-loss_render1.0_splatter1.0_lpips1.0-lr1e-05-Plat5/eval_epoch_30/model.safetensors" \
#  --scheduler_type "cosine"

# ## 4GPUs
# accelerate launch --config_file acc_configs/gpu4.yaml main_zero123plus_v4_batch_code_unet_rendering_loss.py big --workspace runs/zerp123plus_batch/workspace_debug \
#     --lr 1e-6 --num_epochs 10001 --eval_iter 10 --save_iter 10 --lr_scheduler Plat \
#     --lr_scheduler_patience 5 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 0 \
#     --desc '4gpus-unet-w-rendering-loss' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCodeUnet --decode_splatter_to_128 \
#     --splatter_guidance_interval 1 --save_train_pred -1  --overfit_one_scene \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --resume_unet "runs/zerp123plus_batch/workspace_unet/20240309-4-gpus-only-latent-loss-resume20240308-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0-lr4e-05-Plat5/model.safetensors" \
#     --resume "runs/zerp123plus_batch/workspace_train/20240315-4gpus_decode_splatter_to_128_nearest-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter0.5_lpips1.0-lr0.0002-Plat100/model.safetensors"

# # ## 4GPUs[NEW]
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_unet_rendering_loss.py big --workspace runs/zerp123plus_batch/workspace_ablation \
#     --lr 1e-5 --num_epochs 10001 --eval_iter 20 --save_iter 20 --lr_scheduler Plat \
#     --lr_scheduler_patience 5 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 16 \
#     --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 0   \
#     --desc '1gpus-unet-w-rendering-loss-not_rendering_input_views' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCodeUnet \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --resume_unet "runs/zerp123plus_batch/workspace_ablation/20240322-022336-unet-w-rendering-loss-resume_unet_20240320-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-numV6-loss_render1.0_splatter1.0_lpips1.0-lr1e-05-Plat5/eval_epoch_30/model.safetensors" \
#     --resume "runs/zerp123plus_batch/workspace_ablation/20240314-ablation4-fixed-encode-range-4gpus-resume-e30+180-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0_lpips1.0-lr0.0002-Plat100/eval_epoch_300/model.safetensors"

# # [TRAIN: 3D Diffusion]: diffusion unet is in charge of output multiview images, and the decoder is only decode each feature map to one splatter image
# # ## 1GPU
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_MVunet_rendering_loss.py big --workspace runs/zerp123plus_batch/workspace_debug \
#     --lr 1e-5 --num_epochs 10001 --eval_iter 20 --save_iter 20 --lr_scheduler Plat \
#     --lr_scheduler_patience 5 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 16 \
#     --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 0   \
#     --desc 'mvdiffusion' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianMVDiffusion \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --custom_pipeline ./zero123plus/pipeline_v3_mvunet.py \
#     --codes_from_encoder --scheduler_type "cosine"
    
  
# # [INFERENCE with diffusion]
# # export CUDA_VISIBLE_DEVICES=1

# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference.py big --workspace runs/zerp123plus_batch/workspace_inference \
#     --lr 2e-4 --num_epochs 10001 --eval_iter 10 --save_iter 10 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 0 \
#     --desc 'ablation2_fixed_encode_range_epoch30' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_diffusion \
#     --resume 'runs/zerp123plus_batch/workspace_ablation/20240309-ablation_2_fixed_encode_range_4gpus-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0-lr0.0002-Plat100/model.safetensors'

# # # [INFERENCE to see STAT with diffusion]
# # export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference_stat.py big --workspace runs/zerp123plus_batch/workspace_inference \
#     --lr 2e-4 --num_epochs 10001 --eval_iter 10 --save_iter 10 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 0 \
#     --desc 'latent-stats-save-tensors' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_diffusion --codes_from_cache --codes_from_encoder \
#     --resume 'runs/zerp123plus_batch/workspace_train/20240315-4gpus_decode_splatter_to_128_nearest-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter0.5_lpips1.0-lr0.0002-Plat100/model.safetensors'



# # # # ## Train: optimize latent code and (high res) splatter images
# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code.py big --workspace runs/zerp123plus_batch/workspace_debug_codedir \
#     --lr 2e-4 --num_epochs 10001 --eval_iter 100 --save_iter 100 --lr_scheduler Plat --lr_scheduler_patience 5 \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 0 \
#     --desc 'optimize-code-not-splatter-w-guidance-large-code-lr' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --overfit_one_scene
#     # \
    # --resume runs/zerp123plus_batch/workspace_train/20240301-001854-optimize-code-splatter-v0_unfreeze_all-skip_predict_x0-loss_render1.0_splatter1.0-lr0.0002-Plat5/model.safetensors

# # # # # ### gpus=2
# # accelerate launch --main_process_port 29514 --config_file acc_configs/gpu2.yaml main_zero123plus_v4_batch_code.py big --workspace runs/zerp123plus_batch/workspace_debug_codedir \
# #     --lr 2e-4 --num_epochs 10001 --eval_iter 5 --save_iter 5 --lr_scheduler Plat --lr_scheduler_patience 5 \
# #     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
# #     --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 
# 0 --lambda_lpips 0 \
# #     --desc 'optimize-code-2gpus' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
# #     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
# #     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
# #     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode --save_train_pred -1
    
# debug training
# accelerate launch --config_file acc_configs/gpu1.yaml main.py big --workspace workspace_ft --resume pretrained/model_fp16.safetensors --num_epochs 1000
# accelerate launch --config_file acc_configs/gpu1.yaml main.py big --workspace workspace_debug

# # training (should use slurm)
# accelerate launch --config_file acc_configs/gpu8.yaml main.py big --workspace workspace

# # test
# python infer_zero123plus.py big --workspace runs/zerp123plus_overfit/workspace_test --resume pretrained/model_fp16.safetensors --test_path ${DATA_DIR_HYDRANT} --num_input_views 6
# python infer_debug.py big --workspace runs/LGM_optimize_splatter/workspace_test_debug_ply --resume pretrained/model_fp16.safetensors --test_path ${DATA_DIR_HYDRANT} --num_input_views 6
# python infer.py big --workspace workspace_test --resume workspace/model.safetensors --test_path data_test
# python infer_cw.py big --workspace workspace_test_debug_cam --resume pretrained/model_fp16.safetensors --test_path '/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/0a9b36d36e904aee8b51e978a7c0acfd' --num_input_views 6

# # gradio app
# python app.py big --resume workspace/model.safetensors

# # local gui
# python gui.py big --output_size 800 --test_path workspace_test/anya_rgba.ply

# # mesh conversion
# python convert.py big --test_path workspace_test/anya_rgba.ply
