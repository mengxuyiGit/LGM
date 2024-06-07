DATA_RENDERING_ROOT_LVIS_46K=/mnt/lingjie_cache/lvis_dataset/testing
DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT=/mnt/lingjie_cache/lvis_splatters/testing
DATA_RENDERING_ROOT_LVIS_46K_CLUSTER=/home/chenwang/data/lvis_dataset/testing
DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_CLUSTER=/mnt/kostas-graid/sw/envs/xuyimeng/Repo/zero-1-to-G/runs/lvis/workspace_test/testing/1000-1999/20240606-154839-fov60-opt_3channel-no_splatter_loss-random_bg-loss_render1.0_lpips1.0-lr0.001-/splatters_mv_inference
# /mnt/kostas-graid/sw/envs/xuyimeng/Repo/zero-1-to-G/runs/lvis/workspace_test/testing/1000-1999/20240606-160336-fov60-opt_3channel_with_CLIP-no_splatter_loss-random_bg-loss_render1.0_lpips1.0-lr0.001-/splatters_mv_inference
# /mnt/kostas-graid/sw/envs/xuyimeng/Repo/zero-1-to-G/runs/lvis/workspace_test/testing/1000-1999/20240606-151858-fov60-opt_3channel-no_splatter_loss-random_bg-loss_render1.0_lpips1.0-lr0.001-/splatters_mv_inference
# /mnt/kostas-graid/sw/envs/xuyimeng/Repo/zero-1-to-G/runs/lvis/workspace_debug/testing/1000-1999/20240606-151405-fov60-debug_3channel-loss_render1.0_lpips1.0-lr0.001-/splatters_mv_inference
# /mnt/kostas-graid/sw/envs/xuyimeng/Repo/zero-1-to-G/runs/lvis/workspace_test/testing/1000-1999/20240606-071224-fov60-LargeLR2e-3-splatter_lpips0.1-random_bg-loss_render1.0_lpips1.0-lr0.002-/splatters_mv_inference
# /mnt/kostas-graid/sw/envs/xuyimeng/Repo/zero-1-to-G/runs/lvis/workspace_test/testing/1000-1999/20240606-070846-fov60-splatter_lpips0.1-random_bg-loss_render1.0_lpips1.0-lr0.001-/splatters_mv_inference
# /mnt/kostas-graid/sw/envs/xuyimeng/Repo/zero-1-to-G/runs/lvis/workspace_test/testing/1000-1999/20240606-060337-fov60-splatter_lpips0.1-random_bg-loss_render1.0_lpips1.0-lr0.001-/splatters_mv_inference
# /mnt/kostas-graid/sw/envs/xuyimeng/Repo/zero-1-to-G/runs/lvis/workspace_test/testing/1000-1999/20240606-051032-fov60-random_bg-splatter_mse-loss_render1.0_lpips1.0-lr0.001-/splatters_mv_inference
# /mnt/kostas-graid/sw/envs/xuyimeng/Repo/zero-1-to-G/runs/lvis/workspace_test/testing/1000-1999/20240606-022318-fov60-random_bg-tv_loss0.1-loss_render1.0_lpips1.0-lr0.001-/splatters_mv_inference
# /mnt/kostas-graid/sw/envs/xuyimeng/Data/lvis/data_processing/testing
# /mnt/kostas-graid/sw/envs/xuyimeng/Repo/zero-1-to-G/runs/lvis/workspace_test/testing/1000-1999/20240605-184259-bsz4_fov60_find_lr-loss_render1.0_lpips1.0-lr0.001-/splatters_mv_inference
# /mnt/kostas-graid/sw/envs/xuyimeng/Repo/zero-1-to-G/runs/lvis/workspace_/testing/1000-1999/20240606-011101-fov60-tv_loss-loss_render1.0_lpips1.0-lr0.001-/splatters_mv_inference
# /mnt/kostas-graid/sw/envs/xuyimeng/Repo/zero-1-to-G/runs/lvis/workspace_/testing/1000-1999/20240605-230333-fov60-random_bg-loss_render1.0_lpips1.0-lr0.001-/splatters_mv_inference
# 
# /mnt/kostas-graid/sw/envs/xuyimeng/Repo/zero-1-to-G/runs/lvis/workspace_test/testing/1000-1999/20240605-183147-bsz4_fov60_find_lr-loss_render1.0_lpips1.0-lr0.005-


# export CUDA_VISIBLE_DEVICES=0,1,2,3
# accelerate launch --main_process_port 29514 --config_file acc_configs/gpu4.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_accumulate.py big \
#     --workspace runs/finetune_decoder/workspace_train \
#     --lr 1e-5 --max_train_steps 30000 --eval_iter 100 --save_iter 100 --lr_scheduler Plat \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0.1 --lambda_rendering 10 --lambda_alpha 0 --lambda_lpips 10 \
#     --desc 'weighted_splatter_loss-render_lossx10_resume007_1400steps-4gpus_bsz2_accumulate16' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
#     --set_random_seed --batch_size 2 --num_workers 1 \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --only_train_attention --rendering_loss_use_weight_t \
#     --finetune_decoder --gradient_accumulation_steps 16 --output_size 320 \
#     --lambda_each_attribute_loss 1 1. 1 1 10 \
#     --resume_decoder /mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_decoder/workspace_train/00007-resume_smallLR_render_lossx10_splatter700steps_4gpus_bsz2_accumulate32-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render10.0_splatter0.1_lpips10.0-lr1e-05-Plat50/eval_global_step_1400_ckpt/model.safetensors
#     # --resume_decoder /mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_decoder/workspace_train/00005-resume_decoder_1200steps_w_splatter_loss_4gpus_bsz2_accumulate32-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render0.1_splatter5.0_lpips0.1-lr0.0001-Plat50/eval_global_step_700_ckpt/model.safetensors

# # [June 03 Decoder with domain embedding]
# # [no rendering loss]
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# accelerate launch --main_process_port 29514 --config_file acc_configs/gpu4.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_accumulate.py big \
#     --workspace runs/finetune_decoder/workspace_train \
#     --lr 1e-6 --num_epochs 10001 --eval_iter 500 --save_iter 500 --lr_scheduler Plat \
#     --lr_scheduler_patience 5 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 10 --lambda_rendering 0 --lambda_alpha 0 --lambda_lpips 0 \
#     --desc 'decoder-learnable_domain_embedding-no_rendering_loss-resume10500-smallerLR' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
#     --set_random_seed --batch_size 2 --num_workers 2 \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --only_train_attention --rendering_loss_use_weight_t \
#     --finetune_decoder --gradient_accumulation_steps 2 --decoder_with_domain_embedding --decoder_domain_embedding_mode learnable \
#     --resume_decoder /mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_decoder/workspace_train/00009-decoder-add_learnable_zero_init_domain_embedding-no_rendering_loss-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_splatter10.0-lr0.0001-Plat5/eval_global_step_10500_ckpt/model.safetensors

# # # [with rendering loss]
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# accelerate launch --main_process_port 29514 --config_file acc_configs/gpu4.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_accumulate.py big \
#     --workspace runs/finetune_decoder/workspace_train \
#     --lr 1e-5 --num_epochs 10001 --eval_iter 500 --save_iter 500 --lr_scheduler Plat \
#     --lr_scheduler_patience 5 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 10 --lambda_rendering 0 --lambda_alpha 0 --lambda_lpips 0 \
#     --desc 'decoder-no_domain_embedding-no_rendering_loss-resume10500-smallerLR' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
#     --set_random_seed --batch_size 2 --num_workers 2 \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --only_train_attention --rendering_loss_use_weight_t \
#     --finetune_decoder --gradient_accumulation_steps 2 \
#     --resume_decoder /mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_decoder/workspace_train/00012-decoder-no_domain_embedding-no_rendering_loss-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_splatter10.0-lr0.0001-Plat5/eval_global_step_10500_ckpt/model.safetensors
# # --decoder_with_domain_embedding

# singleGPU
export CUDA_VISIBLE_DEVICES=1
accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_accumulate.py big \
    --workspace runs/finetune_decoder/workspace_debug \
    --lr 1e-4 --num_epochs 10001 --eval_iter 200 --save_iter 200 --lr_scheduler Plat \
    --lr_scheduler_patience 5 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
    --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
    --lambda_splatter 10 --lambda_rendering 0.1 --lambda_alpha 0 --lambda_lpips 0.1 \
    --desc 'load_normed_splatter_mv1999-fov60-random_bg' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K_CLUSTER} \
    --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_CLUSTER} \
    --set_random_seed --batch_size 1 --num_workers 1 \
    --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
    --scale_clamp_max -2 --scale_clamp_min -10 \
    --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
    --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
    --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
    --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
    --bg 1.0 --fovy 60 --only_train_attention --rendering_loss_use_weight_t \
    --finetune_decoder --gradient_accumulation_steps 1 --splatter_mv_already_normalized


# # singleGPU
# export CUDA_VISIBLE_DEVICES=4
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_accumulate.py big \
#     --workspace runs/finetune_decoder/workspace_high_freq \
#     --lr 1e-4 --num_epochs 10001 --eval_iter 20 --save_iter 20 --lr_scheduler Plat \
#     --lr_scheduler_patience 5 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 10 --lambda_rendering 0.1 --lambda_alpha 0 --lambda_lpips 0.1 \
#     --desc 'compare_splatters-fov60_lr1e-3_iter3k-20240605-183147' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K_CLUSTER} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_CLUSTER} \
#     --set_random_seed --batch_size 1 --num_workers 2 \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --only_train_attention --rendering_loss_use_weight_t \
#     --finetune_decoder --gradient_accumulation_steps 1 
#     # --decoder_with_domain_embedding --decoder_domain_embedding_mode mlp
#     # --decoder_with_domain_embedding
#     #  --decoder_not_use_rendering_loss
#     # --skip_training  --lambda_each_attribute_loss 1 1. 1 1 10 \
