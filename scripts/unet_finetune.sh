# [MAY 23] Parallel with finetune decoder, same training code, controled with flag
DATA_RENDERING_ROOT_LVIS_46K=/mnt/lingjie_cache/lvis_dataset/testing
DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT=/mnt/lingjie_cache/lvis_splatters/testing

# export CUDA_VISIBLE_DEVICES=4,5,6,7
# accelerate launch --main_process_port 29516 --config_file acc_configs/gpu4.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
#     --workspace runs/finetune_unet/workspace_train \
#     --lr 1e-6 --max_train_steps 30000 --eval_iter 100 --save_iter 100 --lr_scheduler Plat \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'train_unet_smallerBSZ_resume05_2100_steps-4gpus_bsz2_accumulate32' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
#     --set_random_seed --batch_size 2 --num_workers 1 \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --only_train_attention --rendering_loss_use_weight_t \
#     --train_unet --gradient_accumulation_steps 16 --output_size 320 \
#     --resume_unet /mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_unet/workspace_train/00005-train_unet_smallLR_resume_1000steps-4gpus_bsz2_accumulate32-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-train_unet-skip_predict_x0-loss_render1.0_lpips1.0-lr1e-06-Plat50/eval_global_step_2100_ckpt/model.safetensors
#     # --resume_unet /mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_unet/workspace_train/00002-train_unet-4gpus_bsz2_accumulate32-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-train_unet-skip_predict_x0-loss_render1.0_lpips1.0-lr3e-05-Plat50/eval_global_step_1000_ckpt/model.safetensors

# # [MAY 25] Finetune unet with single domain
# # export CUDA_VISIBLE_DEVICES=4,5,6,7
# # This training script works!! (May 26 checked results)
# accelerate launch --main_process_port 29516 --config_file acc_configs/gpu6.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
#     --workspace runs/finetune_unet/workspace_train \
#     --lr 3e-5 --max_train_steps 30000 --eval_iter 100 --save_iter 100 --lr_scheduler Plat \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'train_unet_single_attr_RGBs_6gpus_bsz12_accumulate8' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
#     --set_random_seed --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v2.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --only_train_attention --rendering_loss_use_weight_t \
#     --train_unet --train_unet_single_attr "rgbs" \
#     --batch_size 12 --num_workers 1 --gradient_accumulation_steps 6

# # [MAY 26] Finetune unet with single domain RGB, smaller Bsc
# export CUDA_VISIBLE_DEVICES=0,1
# accelerate launch --main_process_port 29516 --config_file acc_configs/gpu2.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
#     --workspace runs/finetune_unet/workspace_train \
#     --lr 3e-5 --max_train_steps 30000 --eval_iter 100 --save_iter 100 --lr_scheduler Plat \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'train_unet_single_attr_RGBs_smallBSZ_2gpus_bsz12_accumulate2' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
#     --set_random_seed --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v2.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --only_train_attention --rendering_loss_use_weight_t \
#     --train_unet --train_unet_single_attr "rgbs" \
#     --batch_size 12 --num_workers 1 --gradient_accumulation_steps 2 

# # [MAY 26] Finetune unet with single domain RGB, smaller Bsc, pipeline:v7
# export CUDA_VISIBLE_DEVICES=2,3,4,5
# accelerate launch --main_process_port 29517 --config_file acc_configs/gpu4.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
#     --workspace runs/finetune_unet/workspace_train \
#     --lr 3e-5 --max_train_steps 30000 --eval_iter 100 --save_iter 100 --lr_scheduler Plat \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'train_unet_pipev7_single_attr_RGBs_smallBSZ_4gpus_bsz12_accumulate2' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
#     --set_random_seed --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --only_train_attention --rendering_loss_use_weight_t \
#     --train_unet --train_unet_single_attr "rgbs" \
#     --batch_size 12 --num_workers 1 --gradient_accumulation_steps 2 


# # [MAY 27] Finetune unet with single domain RGB, smaller Bsc, pipeline:v7 no seq
# export CUDA_VISIBLE_DEVICES=2,3,4,5
# accelerate launch --main_process_port 29517 --config_file acc_configs/gpu4.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
#     --workspace runs/finetune_unet/workspace_train \
#     --lr 3e-5 --max_train_steps 30000 --eval_iter 200 --save_iter 200 --lr_scheduler Plat \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'train_unet_pipev7_no_seq_NO_POS_EMBED_single_attr_RGBs_smallBSZ_4gpus_bsz12_accumulate1' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
#     --set_random_seed --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v7_no_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --only_train_attention --rendering_loss_use_weight_t \
#     --train_unet --train_unet_single_attr "rgbs" \
#     --batch_size 12 --num_workers 1 --gradient_accumulation_steps 1

# # [MAY 27] Finetune unet with single domain RGB, smaller Bsc, pipeline:v7 no seq
# export CUDA_VISIBLE_DEVICES=0,1
# accelerate launch --main_process_port 29514 --config_file acc_configs/gpu2.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
#     --workspace runs/finetune_unet/workspace_train \
#     --lr 3e-5 --max_train_steps 30000 --eval_iter 200 --save_iter 200 --lr_scheduler Plat \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'train_unet_pipev7_no_seq_NO_POS_EMBED_single_attr_OPACITYs_smallBSZ_4gpus_bsz12_accumulate1' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
#     --set_random_seed --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v7_no_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --only_train_attention --rendering_loss_use_weight_t \
#     --train_unet --train_unet_single_attr "opacity" \
#     --batch_size 12 --num_workers 1 --gradient_accumulation_steps 2

# # [MAY 27] Finetune unet with single domain RGB, smaller Bsc, pipeline:v7 no seq
# export CUDA_VISIBLE_DEVICES=0
# accelerate launch --main_process_port 29513 --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
#     --workspace runs/finetune_unet/workspace_train \
#     --lr 3e-5 --max_train_steps 30000 --eval_iter 200 --save_iter 200 --lr_scheduler Plat \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'train_unet_VERYsmallBSZ12_single_attr_OPACITYs_pipev7_no_seq_NO_POS_EMBED_1gpus_bsz12_accumulate1' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
#     --set_random_seed --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v7_no_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --only_train_attention --rendering_loss_use_weight_t \
#     --train_unet --train_unet_single_attr "opacity" \
#     --batch_size 12 --num_workers 1 --gradient_accumulation_steps 1

# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --main_process_port 29512 --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
#     --workspace runs/finetune_unet/workspace_train \
#     --lr 3e-5 --max_train_steps 30000 --eval_iter 200 --save_iter 200 --lr_scheduler Plat \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'train_unet_VERYsmallBSZ12_single_attr_POS_pipev7_no_seq_NO_POS_EMBED_1gpus_bsz12_accumulate1' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
#     --set_random_seed --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v7_no_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --only_train_attention --rendering_loss_use_weight_t \
#     --train_unet --train_unet_single_attr "pos" \
#     --batch_size 12 --num_workers 1 --gradient_accumulation_steps 1


# # [MAY 27] Finetune unet with 2 domains: RGB + SCALE, smaller Bsc, pipeline:v7 with seq reshape
# export CUDA_VISIBLE_DEVICES=2,3,4,5
# accelerate launch --main_process_port 29518 --config_file acc_configs/gpu4.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
#     --workspace runs/finetune_unet/workspace_train \
#     --lr 3e-5 --max_train_steps 30000 --eval_iter 200 --save_iter 200 --lr_scheduler Plat \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'train_unet_pipev7_CD_attn_NO_POS_EMBED_2attr_RGBs_SCALE_smallBSZ_4gpus_bsz6_accumulate1' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
#     --set_random_seed --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --only_train_attention --rendering_loss_use_weight_t \
#     --train_unet --train_unet_single_attr "rgbs" "scale" \
#     --batch_size 6 --num_workers 1 --gradient_accumulation_steps 1

# # [MAY 27] Finetune unet with 2 domains: RGB + SCALE, smaller Bsc, pipeline:v7 with seq reshape
# export CUDA_VISIBLE_DEVICES=2,3,4,5
# accelerate launch --main_process_port 29518 --config_file acc_configs/gpu4.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
#     --workspace runs/finetune_unet/workspace_train \
#     --lr 3e-5 --max_train_steps 30000 --eval_iter 200 --save_iter 200 --lr_scheduler Plat \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'train_unet_pipev7_CD_attn_NO_POS_EMBED_2attr_RGBs_SCALE_smallBSZ_4gpus_bsz6_accumulate1' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
#     --set_random_seed --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --only_train_attention --rendering_loss_use_weight_t \
#     --train_unet --train_unet_single_attr "rgbs" "scale" \
#     --batch_size 6 --num_workers 1 --gradient_accumulation_steps 1

# [MAY 27] Finetune unet with ALL domains, smaller Bsc, pipeline:v7 with seq reshape
export CUDA_VISIBLE_DEVICES=2,3,4,5
accelerate launch --main_process_port 29518 --config_file acc_configs/gpu4.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
    --workspace runs/finetune_unet/workspace_train \
    --lr 3e-5 --max_train_steps 30000 --eval_iter 200 --save_iter 200 --lr_scheduler Plat \
    --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
    --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
    --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
    --desc 'train_unet_ALL_ATTR_pipev7_CD_attn_no_pos_embed_smallBSZ_4gpus_bsz2_accumulate2' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
    --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
    --set_random_seed --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
    --scale_clamp_max -2 --scale_clamp_min -10 \
    --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
    --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
    --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
    --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
    --bg 1.0 --fovy 50 --only_train_attention --rendering_loss_use_weight_t \
    --train_unet \
    --batch_size 2 --num_workers 1 --gradient_accumulation_steps 2


# [DEBUG] 
DATA_RENDERING_ROOT_LVIS_46K_CLUSTER=/home/chenwang/data/lvis_dataset/testing
DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_CLUSTER=/mnt/kostas-graid/sw/envs/xuyimeng/Data/lvis/data_processing/testing

# accelerate launch --main_process_port 29516 --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
#     --workspace runs/finetune_unet/workspace_debug \
#     --lr 3e-5 --max_train_steps 30000 --eval_iter 100 --save_iter 100 --lr_scheduler Plat \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'debug_v7_class_labels' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K_CLUSTER} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_CLUSTER} \
#     --set_random_seed --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --only_train_attention --rendering_loss_use_weight_t \
#     --train_unet --train_unet_single_attr "rgbs" "scale" \
#     --batch_size 12 --num_workers 1 --gradient_accumulation_steps 2 
#     # --overfit_one_scene

# accelerate launch --main_process_port 29516 --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
#     --workspace runs/finetune_unet/workspace_debug \
#     --lr 3e-5 --max_train_steps 30000 --eval_iter 100 --save_iter 100 --lr_scheduler Plat \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'debug_v7_all_attr' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K_CLUSTER} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_CLUSTER} \
#     --set_random_seed --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --only_train_attention --rendering_loss_use_weight_t \
#     --train_unet \
#     --batch_size 12 --num_workers 1 --gradient_accumulation_steps 2 --overfit_one_scene
# # --train_unet_single_attr "rgbs" "scale" 

# # [MAY 27] Finetune unet with single domain RGB, smaller Bsc, pipeline:v7 no seq
# accelerate launch --main_process_port 29516 --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
#     --workspace runs/finetune_unet/workspace_debug \
#     --lr 3e-5 --max_train_steps 30000 --eval_iter 100 --save_iter 100 --lr_scheduler Plat \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'debug+_v7_no_seq_no_pos_embed_reshape' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K_CLUSTER} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_CLUSTER} \
#     --set_random_seed --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v7_no_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --only_train_attention --rendering_loss_use_weight_t \
#     --train_unet --train_unet_single_attr "rgbs" \
#     --batch_size 12 --num_workers 1 --gradient_accumulation_steps 2 --overfit_one_scene
