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

# # [JUN 6] Finetune unet ALL LAYERS, all domains, pipeline:v7
# accelerate launch --main_process_port 29517 --config_file acc_configs/gpu4.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
#     --workspace runs/finetune_unet/workspace_train \
#     --lr 1e-5 --max_train_steps 30000 --eval_iter 200 --save_iter 200 --lr_scheduler Plat \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1  --lambda_splatter_lpips 0 \
#     --desc 'xyz_t=10-only_train_attention-v7' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
#     --set_random_seed --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --rendering_loss_use_weight_t \
#     --train_unet --only_train_attention --xyz_zero_t \
#     --batch_size 2 --num_workers 1 --gradient_accumulation_steps 1 


# # [JUN 7] Add rendering loss
# accelerate launch --main_process_port 29517 --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
#     --workspace runs/finetune_unet/workspace_debug \
#     --lr 1e-5 --max_train_steps 30000 --eval_iter 500 --save_iter 1000 --lr_scheduler Plat \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1  --lambda_splatter_lpips 0 \
#     --desc 'rendering_loss-only_train_attention-v7-log_dirty_data' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
#     --set_random_seed --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --rendering_loss_use_weight_t \
#     --train_unet --only_train_attention \
#     --batch_size 1 --num_workers 1 --gradient_accumulation_steps 1 

# # [JUN 7] Add rendering loss
# export CUDA_VISIBLE_DEVICES=0
# accelerate launch --main_process_port 29518 --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
#     --workspace runs/finetune_unet/workspace_debug \
#     --lr 1e-5 --max_train_steps 30001 --eval_iter 400 --save_iter 400 --lr_scheduler Plat \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1  --lambda_splatter_lpips 0 \
#     --desc 'debug_sample_hidden_states' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
#     --set_random_seed --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --rendering_loss_use_weight_t \
#     --train_unet --only_train_attention \
#     --batch_size 1 --num_workers 1 --gradient_accumulation_steps 2 
    
#     # \
#     # --resume_decoder /mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_decoder/workspace_train/00007-resume_smallLR_render_lossx10_splatter700steps_4gpus_bsz2_accumulate32-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render10.0_splatter0.1_lpips10.0-lr1e-05-Plat50/eval_global_step_1400_ckpt/model.safetensors \
#     # --resume_unet /mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_unet/workspace_train/20240610-135255-DOMAIN_EMBED_CAT_resume400-fted_decoeder-rendering_mse_lpips_loss-only_train_attention-v7-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-train_unet-loss_render1.0_lpips1.0-lr1e-05-Plat50/eval_global_step_8000_ckpt/model.safetensors
#     # # /mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_unet/workspace_train/20240565-DOMAIN_EMBED_CAT_resume400-fted_decoeder-rendering_mse_lpips_loss-only_train_attention-v7-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-train_unet-numV10-loss_render1.0_lpips1.0-lr1e-05-Plat50/eval_global_step_800_ckpt/model.safetensors
#     # # /mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_unet/workspace_train/20240564-DOMAIN_EMBED_CAT-fted_decoeder-rendering_mse_lpips_loss-only_train_attention-v7-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-train_unet-numV10-loss_render1.0_lpips1.0-lr1e-05-Plat50/eval_global_step_400_ckpt/model.safetensors
#     # # /mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_unet/workspace_train/20240558-rendering_mse_lpips_loss-only_train_attention-v7-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-train_unet-numV10-loss_render1.0_lpips1.0-lr1e-05-Plat50/eval_global_step_29000_ckpt/model.safetensors
    
# # [Jun 11] Not use prior: train all layers from scratch. pipe: v8.
# # batch size: 32
# # export CUDA_VISIBLE_DEVICES=2,3,4,5
# accelerate launch --main_process_port 29518 --config_file acc_configs/gpu8.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
#     --workspace runs/finetune_unet/workspace_train_june \
#     --lr 3e-5 --max_train_steps 30000 --eval_iter 500 --save_iter 500 --lr_scheduler Plat \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 0 --lambda_alpha 0 --lambda_lpips 0  --lambda_splatter_lpips 0 \
#     --desc 'train_random_init_unet-all_layer-resume5500-pipev7-bsz32' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
#     --set_random_seed --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --rendering_loss_use_weight_t \
#     --train_unet --random_init_unet \
#     --batch_size 1 --num_workers 1 --gradient_accumulation_steps 1 \
#     --resume_unet /mnt/kostas_home/lilym/LGM/LGM/runs/finetune_unet/workspace_train_june/00006-train_random_init_unet-all_layer-pipev7-bsz32-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-train_unet-loss-lr3e-05-Plat50/eval_global_step_5500_ckpt/model.safetensors

# # [Junn 12] All tricks together: pipev8, class_emb cat, large bsz, rendering loss, only attn
# accelerate launch --main_process_port 29517 --config_file acc_configs/gpu8.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
#     --workspace runs/finetune_unet/workspace_train_june \
#     --lr 1e-5 --max_train_steps 30000 --eval_iter 500 --save_iter 500 --lr_scheduler Plat \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 0 --lambda_alpha 0 --lambda_lpips 0  --lambda_splatter_lpips 0 \
#     --desc 'train_unet-only_attn-pipev8-bsz32-resume20kckpt-lr1e-5' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
#     --set_random_seed \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v8_cat.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --rendering_loss_use_weight_t \
#     --train_unet --class_emb_cat --only_train_attention \
#     --batch_size 2 --num_workers 1 --gradient_accumulation_steps 2 \
#     --resume_unet /mnt/kostas_home/lilym/LGM/LGM/runs/finetune_unet/workspace_train_june/00007-train_unet-only_attn-pipev8-bsz32-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-train_unet-loss-lr3e-05-Plat50/eval_global_step_20000_ckpt/model.safetensors

# # [Junn 13] All tricks together: pipev8, class_emb cat, large bsz, rendering loss, only attn
# accelerate launch --main_process_port 29517 --config_file acc_configs/gpu8.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
#     --workspace runs/finetune_unet/workspace_train_june \
#     --lr 5e-6 --max_train_steps 30000 --eval_iter 500 --save_iter 500 --lr_scheduler Plat \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 --lambda_splatter_lpips 0 \
#     --desc 'train_unet-rendering_mse_lpips_loss-only_attn-pipev8-bsz16-resume42kckpt-lr5e-6' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
#     --set_random_seed \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v8_cat.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --rendering_loss_use_weight_t \
#     --train_unet --class_emb_cat --only_train_attention \
#     --batch_size 1 --num_workers 1 --gradient_accumulation_steps 2 \
#     --resume_decoder /mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_decoder/workspace_train/00007-resume_smallLR_render_lossx10_splatter700steps_4gpus_bsz2_accumulate32-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render10.0_splatter0.1_lpips10.0-lr1e-05-Plat50/eval_global_step_1400_ckpt/model.safetensors \
#     --resume_unet /mnt/kostas_home/lilym/LGM/LGM/runs/finetune_unet/workspace_train_june/00011-train_unet-rendering_mse_lpips_loss-only_attn-pipev8-bsz16-resume28kckpt-lr1e-5-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-train_unet-loss_render1.0_lpips1.0-lr1e-05-Plat50/eval_global_step_6500_ckpt/model.safetensors \
#     --invalid_list /mnt/kostas_home/lilym/LGM/LGM/data_lists/lvis_invalid_uids_nineviews.json \
#     --drop_cond_prob 0.1
#     # /mnt/kostas_home/lilym/LGM/LGM/runs/finetune_unet/workspace_train_june/00008-train_unet-only_attn-pipev8-bsz32-resume20kckpt-lr1e-5-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-train_unet-loss-lr1e-05-Plat50/eval_global_step_8000_ckpt/model.safetensors \

# DATA_RENDERING_ROOT_LVIS_46K_CLUSTER=/home/chenwang/data/lvis_dataset/testing
# DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_CLUSTER=/mnt/kostas-graid/datasets/xuyimeng/lvis/data_processing/testing

# # [Junn 16] NO rendering loss. All tricks together: pipev8, class_emb cat, large bsz, , only attn
# accelerate launch --main_process_port 29517 --config_file acc_configs/gpu8.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
#     --workspace runs/finetune_unet/workspace_train_june \
#     --lr 4e-6 --max_train_steps 100000 --eval_iter 500 --save_iter 500 --lr_scheduler Plat \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 0 --lambda_alpha 0 --lambda_lpips 0 --lambda_splatter_lpips 0 \
#     --desc 'train_unet-exp_act-with_timeproj_clsemb-resume29k-bsz64-NO-rendering-only_attn-pipev8-resum50kckpt-lr1e-6' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
#     --set_random_seed \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v8_cat.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --rendering_loss_use_weight_t \
#     --train_unet --class_emb_cat --only_train_attention \
#     --batch_size 2 --num_workers 1 --gradient_accumulation_steps 2 \
#     --resume_decoder /mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_decoder/workspace_train/00007-resume_smallLR_render_lossx10_splatter700steps_4gpus_bsz2_accumulate32-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render10.0_splatter0.1_lpips10.0-lr1e-05-Plat50/eval_global_step_1400_ckpt/model.safetensors \
#     --resume_unet /mnt/kostas_home/lilym/LGM/LGM/runs/finetune_unet/workspace_train_june/00016-train_unet-with_timeproj_clsemb-resume28k-bsz64-NO-rendering-only_attn-pipev8-resum50kckpt-lr1e-6-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-train_unet-loss-lr1e-06-Plat50/eval_global_step_1000_ckpt/model.safetensors \
#     --invalid_list /mnt/kostas_home/lilym/LGM/LGM/data_lists/lvis_invalid_uids_nineviews.json \
#     --drop_cond_prob 0.1

# # [DEBUG] 
# DATA_RENDERING_ROOT_LVIS_46K_CLUSTER=/home/chenwang/data/lvis_dataset/testing
# DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_CLUSTER=/mnt/kostas-graid/datasets/xuyimeng/lvis/data_processing/testing

# # [Junn 16] NO rendering loss. All tricks together: pipev8, class_emb cat, large bsz, , only attn
# accelerate launch --main_process_port 29518 --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
#     --workspace runs/finetune_unet/workspace_debug \
#     --lr 4e-6 --max_train_steps 100000 --eval_iter 500 --save_iter 500 --lr_scheduler Plat \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 0 --lambda_alpha 0 --lambda_lpips 0 --lambda_splatter_lpips 0 \
#     --desc 'train_unet-overfit-from_pretrained' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K_CLUSTER} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_CLUSTER} \
#     --set_random_seed \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v8_cat.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --rendering_loss_use_weight_t \
#     --train_unet --class_emb_cat --only_train_attention \
#     --batch_size 2 --num_workers 1 --gradient_accumulation_steps 1 \
#     --resume_decoder /mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_decoder/workspace_train/00007-resume_smallLR_render_lossx10_splatter700steps_4gpus_bsz2_accumulate32-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render10.0_splatter0.1_lpips10.0-lr1e-05-Plat50/eval_global_step_1400_ckpt/model.safetensors \
#     --resume_unet /mnt/kostas_home/lilym/LGM/LGM/runs/finetune_unet/workspace_train_june/00016-train_unet-with_timeproj_clsemb-resume28k-bsz64-NO-rendering-only_attn-pipev8-resum50kckpt-lr1e-6-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-train_unet-loss-lr1e-06-Plat50/eval_global_step_1000_ckpt/model.safetensors \
#     --invalid_list /mnt/kostas_home/lilym/LGM/LGM/data_lists/lvis_invalid_uids_nineviews.json \
#     --drop_cond_prob 0.1 --overfit_one_scene

# # [Junn 13] Filter invalid objects
# accelerate launch --main_process_port 29517 --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
#     --workspace runs/finetune_unet/workspace_debug \
#     --lr e-5 --max_train_steps 30000 --eval_iter 500 --save_iter 500 --lr_scheduler Plat \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1  --lambda_splatter_lpips 0 \
#     --desc 'CFG_training' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K_CLUSTER} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_CLUSTER} \
#     --set_random_seed \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v8_cat.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --rendering_loss_use_weight_t \
#     --train_unet --class_emb_cat --only_train_attention --drop_cond_prob 0.1 \
#     --batch_size 1 --num_workers 1 --gradient_accumulation_steps 2 \
#     --invalid_list /mnt/kostas_home/lilym/LGM/LGM/data_lists/lvis_invalid_uids_nineviews.json \
#     --resume_unet /mnt/kostas_home/lilym/LGM/LGM/runs/finetune_unet/workspace_train_june/00008-train_unet-only_attn-pipev8-bsz32-resume20kckpt-lr1e-5-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-train_unet-loss-lr1e-05-Plat50/eval_global_step_8000_ckpt/model.safetensors \
    
    

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