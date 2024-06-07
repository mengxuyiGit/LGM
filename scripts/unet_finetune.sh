# [MAY 23] Parallel with finetune decoder, same training code, controled with flag
DATA_RENDERING_ROOT_LVIS_46K=/mnt/lingjie_cache/lvis_dataset/testing
DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT=/mnt/lingjie_cache/lvis_splatters/testing
    

# # [MAY 30] Not use zero123++ weights, but load SD weights for finetuning
# export CUDA_VISIBLE_DEVICES=0,1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
#     --workspace runs/finetune_SD_unet/workspace_train \
#     --lr 3e-5 --max_train_steps 30000 --eval_iter 400 --save_iter 200 --lr_scheduler Plat \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 --lambda_splatter_lpips 0 \
#     --desc 'load_SD_weights_46K_pipev7_onl_attn' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
#     --set_random_seed --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --rendering_loss_use_weight_t \
#     --train_unet --load_SD_weights --train_unet_single_attr "rgbs" --only_train_attention \
#     --batch_size 12 --num_workers 1 --gradient_accumulation_steps 1

# # [MAY 30] Not use zero123++ weights, but load SD weights for finetuning
# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
#     --workspace runs/finetune_SD_unet/workspace_train \
#     --lr 3e-5 --max_train_steps 30000 --eval_iter 400 --save_iter 200 --lr_scheduler Plat \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 --lambda_splatter_lpips 0 \
#     --desc 'load_SD_weights_46K_pipev7_ALL_LAYER' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
#     --set_random_seed --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --rendering_loss_use_weight_t \
#     --train_unet --load_SD_weights --train_unet_single_attr "rgbs" \
#     --batch_size 6 --num_workers 1 --gradient_accumulation_steps 2

# # [MAY 30] Not doing seq cut, attend all domain & all views together
# export CUDA_VISIBLE_DEVICES=2,3,4,5
# accelerate launch --main_process_port 29517 --config_file acc_configs/gpu4.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
#     --workspace runs/finetune_unet/workspace_train \
#     --lr 3e-5 --max_train_steps 30000 --eval_iter 400 --save_iter 200 --lr_scheduler Plat \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1  --lambda_splatter_lpips 0 \
#     --desc 'train_unet_pipev8_only_attn' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
#     --set_random_seed --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v8_cat.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --rendering_loss_use_weight_t \
#     --train_unet --only_train_attention \
#     --batch_size 2 --num_workers 1 --gradient_accumulation_steps 1

# [JUN 6] Finetune unet ALL LAYERS, all domains, pipeline:v7
accelerate launch --main_process_port 29517 --config_file acc_configs/gpu4.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
    --workspace runs/finetune_unet/workspace_train \
    --lr 1e-5 --max_train_steps 30000 --eval_iter 200 --save_iter 200 --lr_scheduler Plat \
    --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
    --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
    --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1  --lambda_splatter_lpips 0 \
    --desc 'xyz_t=10-only_train_attention-v7' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
    --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
    --set_random_seed --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
    --scale_clamp_max -2 --scale_clamp_min -10 \
    --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
    --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
    --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
    --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
    --bg 1.0 --fovy 50 --rendering_loss_use_weight_t \
    --train_unet --only_train_attention --xyz_zero_t \
    --batch_size 2 --num_workers 1 --gradient_accumulation_steps 1 
    
# [DEBUG] 
DATA_RENDERING_ROOT_LVIS_46K_CLUSTER=/home/chenwang/data/lvis_dataset/testing
DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_CLUSTER=/mnt/kostas-graid/sw/envs/xuyimeng/Data/lvis/data_processing/testing

# # [MAY 29] Finetune unet with ALL domains, with SPLATTER_LOSS + splatter_LPIPS loss, pipeline:v7 with seq reshape
# # export CUDA_VISIBLE_DEVICES=2,3,4,5
# accelerate launch --main_process_port 29519 --config_file acc_configs/gpu4.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
#     --workspace runs/finetune_unet/workspace_train \
#     --lr 3e-5 --max_train_steps 30000 --eval_iter 200 --save_iter 200 --lr_scheduler Plat \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 --lambda_splatter_lpips 0 \
#     --desc 'train_unet_ALL_LAYERS_1kscene_pipev7_4gpus_bsz2_accumulate1' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K_CLUSTER} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_CLUSTER} \
#     --set_random_seed --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --rendering_loss_use_weight_t \
#     --train_unet --train_unet_single_attr "rgbs" \
#     --batch_size 12 --num_workers 1 --gradient_accumulation_steps 1

# # [MAY 30] Finetune unet ALL LAYERS, all domains, pipeline:v7
# accelerate launch --main_process_port 29517 --config_file acc_configs/gpu4.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
#     --workspace runs/finetune_unet/workspace_train \
#     --lr 3e-5 --max_train_steps 30000 --eval_iter 200 --save_iter 200 --lr_scheduler Plat \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1  --lambda_splatter_lpips 0 \
#     --desc 'train_unet_CLUSTER1k_All_Layer_All_Attr_pipev7_seq_4gpus_bsz2_accumulate1' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K_CLUSTER} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_CLUSTER} \
#     --set_random_seed --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --rendering_loss_use_weight_t \
#     --train_unet \
#     --batch_size 2 --num_workers 1 --gradient_accumulation_steps 1


# # [JUN 6] Finetune unet ALL LAYERS, all domains, pipeline:v7
# accelerate launch --main_process_port 29517 --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
#     --workspace runs/finetune_unet/workspace_debug \
#     --lr 1e-5 --max_train_steps 30000 --eval_iter 200 --save_iter 200 --lr_scheduler Plat \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1  --lambda_splatter_lpips 0 \
#     --desc 'debug' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K_CLUSTER} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_CLUSTER} \
#     --set_random_seed --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --rendering_loss_use_weight_t \
#     --train_unet --only_train_attention --xyz_zero_t \
#     --batch_size 2 --num_workers 1 --gradient_accumulation_steps 1 
#     #  --different_t_schedule 0 10 20 30 40