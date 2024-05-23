# [MAY 23] Parallel with finetune decoder, same training code, controled with flag
DATA_RENDERING_ROOT_LVIS_46K=/mnt/lingjie_cache/lvis_dataset/testing
DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT=/mnt/lingjie_cache/lvis_splatters/testing

export CUDA_VISIBLE_DEVICES=4,5,6,7
accelerate launch --main_process_port 29516 --config_file acc_configs/gpu4.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
    --workspace runs/finetune_unet/workspace_train \
    --lr 3e-5 --max_train_steps 30000 --eval_iter 100 --save_iter 100 --lr_scheduler Plat \
    --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
    --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
    --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
    --desc 'train_unet_resume_1000steps-4gpus_bsz2_accumulate32' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
    --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
    --set_random_seed --batch_size 2 --num_workers 1 \
    --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
    --scale_clamp_max -2 --scale_clamp_min -10 \
    --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
    --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
    --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
    --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
    --bg 1.0 --fovy 50 --only_train_attention --rendering_loss_use_weight_t \
    --train_unet --gradient_accumulation_steps 32 --output_size 320 \
    --resume_unet /mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_unet/workspace_train/00002-train_unet-4gpus_bsz2_accumulate32-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-train_unet-skip_predict_x0-loss_render1.0_lpips1.0-lr3e-05-Plat50/eval_global_step_1000_ckpt/model.safetensors

# # singleGPU debug
# # export CUDA_VISIBLE_DEVICES=5
# DATA_RENDERING_ROOT_LVIS_46K_CLUSTER=/home/chenwang/data/lvis_dataset/testing
# DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_CLUSTER=/mnt/kostas-graid/sw/envs/xuyimeng/Data/lvis/data_processing/testing
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
#     --workspace runs/finetune_unet/workspace_debug \
#     --lr 3e-5 --num_epochs 10001 --eval_iter 10 --save_iter 10 --lr_scheduler Plat \
#     --lr_scheduler_patience 5 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'debug_resume_unet_shared_decoder_code' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K_CLUSTER} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_CLUSTER} \
#     --set_random_seed --batch_size 1 --num_workers 2 \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --only_train_attention --rendering_loss_use_weight_t \
#     --train_unet --gradient_accumulation_steps 5 --output_size 320 \
#     --resume_unet /mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_unet/workspace_train/00002-train_unet-4gpus_bsz2_accumulate32-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-train_unet-skip_predict_x0-loss_render1.0_lpips1.0-lr3e-05-Plat50/eval_global_step_1000_ckpt/model.safetensors
#     # --skip_training



# # [MAY 19] finetune original SD-v2 without module change, with batch size accumulation
# DATA_DIR_BATCH_RENDERING='/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999'
# DATA_DIR_BATCH_SPLATTER_GT_ROOT='/home/xuyimeng/Repo/LGM/runs/splatter_gt'
# DATA_DIR_BATCH_VAE_SPLATTER_ROOT="/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold/workspace_optimize"

# accelerate launch --main_process_port 29514 --config_file acc_configs/gpu8.yaml main_zero123plus_v5_batch_marigold_unet_accumulate.py big \
#     --workspace runs/finetune_unet/workspace_train \
#     --lr 3e-5 --eval_iter 20 --save_iter 20 --lr_scheduler Plat --min_lr_scheduled 1e-5 \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc '8gpu-acc16-single-domain' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain --data_path_vae_splatter ${DATA_DIR_BATCH_VAE_SPLATTER_ROOT} \
#     --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --train_unet --only_train_attention \
#     --attribute_subset "rgbs" --gradient_accumulation_steps 16 \
#     --max_train_steps 10000 
#     # --overfit_one_scene 

### [APR 26 - Use fixed rotation] Fast fitting - Batch process


# # [MAY 19] use LVIS 40k data: cw 6 views -> LGM init splatter
# accelerate launch --main_process_port 29514 --config_file acc_configs/gpu8.yaml main_zero123plus_v5_batch_marigold_unet_accumulate.py big \
#     --workspace runs/finetune_unet/workspace_train \
#     --lr 3e-5 --eval_iter 20 --save_iter 20 --lr_scheduler Plat --min_lr_scheduled 1e-5 \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc '8gpu-acc16-single-domain' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain --data_path_vae_splatter ${DATA_DIR_BATCH_VAE_SPLATTER_ROOT} \
#     --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --train_unet --only_train_attention \
#     --attribute_subset "rgbs" --gradient_accumulation_steps 16 \
#     --max_train_steps 10000 
#     # --overfit_one_scene 

