DATA_RENDERING_ROOT_LVIS_46K=/mnt/lingjie_cache/lvis_dataset/testing
DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT=/mnt/lingjie_cache/lvis_splatters/testing
DATA_RENDERING_ROOT_LVIS_46K_CLUSTER=/home/chenwang/data/lvis_dataset/testing
DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_CLUSTER=/mnt/kostas-graid/sw/envs/xuyimeng/Data/lvis/data_processing/testing

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# accelerate launch --main_process_port 29514 --config_file acc_configs/gpu4.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_accumulate.py big \
#     --workspace runs/finetune_decoder/workspace_train \
#     --lr 1e-5 --max_train_steps 30000 --eval_iter 100 --save_iter 100 --lr_scheduler Plat \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0.1 --lambda_rendering 10 --lambda_alpha 0 --lambda_lpips 10 \
#     --desc 'resume_smallLR_render_lossx10_splatter700steps_4gpus_bsz2_accumulate32' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
#     --set_random_seed --batch_size 2 --num_workers 1 \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --only_train_attention --rendering_loss_use_weight_t \
#     --finetune_decoder --gradient_accumulation_steps 32 --output_size 320 \
#     --resume_decoder /mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_decoder/workspace_train/00005-resume_decoder_1200steps_w_splatter_loss_4gpus_bsz2_accumulate32-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render0.1_splatter5.0_lpips0.1-lr0.0001-Plat50/eval_global_step_700_ckpt/model.safetensors \
#     --resume_unet /mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_unet/workspace_train/00002-train_unet-4gpus_bsz2_accumulate32-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-train_unet-skip_predict_x0-loss_render1.0_lpips1.0-lr3e-05-Plat50/eval_global_step_1000_ckpt/model.safetensors

# # singleGPU
# # export CUDA_VISIBLE_DEVICES=5
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared_inference.py big \
#     --workspace runs/finetune_unet/workspace_inference \
#     --lr 1e-4 --num_epochs 10001 --eval_iter 20 --save_iter 20 --lr_scheduler Plat \
#     --lr_scheduler_patience 5 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 100 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'inference_weigthed_splatter_4900steps' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K_CLUSTER} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_CLUSTER} \
#     --set_random_seed --batch_size 1 --num_workers 2 \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --only_train_attention --rendering_loss_use_weight_t \
#     --inference_finetuned_unet --gradient_accumulation_steps 5 --output_size 320 \
#     --log_each_attribute_loss \
#     --resume_unet /mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_unet/workspace_train/00006-train_unet_smallerBSZ_resume05_2100_steps-4gpus_bsz2_accumulate32-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-train_unet-skip_predict_x0-loss_render1.0_lpips1.0-lr1e-06-Plat50/eval_global_step_3700_ckpt/model.safetensors
#     # --resume_unet /mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_unet/workspace_train/00002-train_unet-4gpus_bsz2_accumulate32-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-train_unet-skip_predict_x0-loss_render1.0_lpips1.0-lr3e-05-Plat50/eval_global_step_1000_ckpt/model.safetensors
#     # --skip_training 
#     # --resume_decoder /mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_decoder/workspace_train/00008-weighted_splatter_loss-render_lossx10_resume007_1400steps-4gpus_bsz2_accumulate16-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render10.0_splatter0.1_lpips10.0-lr1e-05-Plat50/eval_global_step_2200_ckpt/model.safetensors \
#     # --resume_decoder /mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_decoder/workspace_train/00007-resume_smallLR_render_lossx10_splatter700steps_4gpus_bsz2_accumulate32-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render10.0_splatter0.1_lpips10.0-lr1e-05-Plat50/eval_global_step_1200_ckpt/model.safetensors \
#     # --resume_decoder /mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_decoder/workspace_train/00005-resume_decoder_1200steps_w_splatter_loss_4gpus_bsz2_accumulate32-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render0.1_splatter5.0_lpips0.1-lr0.0001-Plat50/eval_global_step_700_ckpt/model.safetensors \


# # [MAY 26] inference on finetuning unet for single attribute image
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared_inference.py big \
#     --workspace runs/finetune_unet/workspace_inference \
#     --lr 1e-4 --num_epochs 10001 --eval_iter 20 --save_iter 20 --lr_scheduler Plat \
#     --lr_scheduler_patience 5 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'inference_pipev7_no_seq_NO_POS_EMBED_steps200_smallBSZ_unet_single_attr_RGBs_seed42' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K_CLUSTER} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_CLUSTER} \
#     --set_random_seed --batch_size 1 --num_workers 2 \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v7_no_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --only_train_attention --rendering_loss_use_weight_t \
#     --inference_finetuned_unet --gradient_accumulation_steps 5 --output_size 320 \
#     --log_each_attribute_loss --train_unet_single_attr "rgbs" --save_cond \
#     --resume_unet /mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_unet/workspace_train/00014-train_unet_pipev7_no_seq_NO_POS_EMBED_single_attr_RGBs_smallBSZ_4gpus_bsz12_accumulate1-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-train_unet-loss_render1.0_lpips1.0-lr3e-05-Plat50/eval_global_step_200_ckpt/model.safetensors
#     # --resume_unet /mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_unet/workspace_train/00011-train_unet_single_attr_RGBs_smallBSZ_2gpus_bsz12_accumulate2-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-train_unet-loss_render1.0_lpips1.0-lr3e-05-Plat50/eval_global_step_3400_ckpt/model.safetensors
#     # --resume_unet /mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_unet/workspace_train/00013-train_unet_pipev7_no_seq_single_attr_RGBs_smallBSZ_4gpus_bsz12_accumulate1-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-train_unet-loss_render1.0_lpips1.0-lr3e-05-Plat50/eval_global_step_3400_ckpt/model.safetensors
#     # --resume_unet /mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_unet/workspace_train/00012-train_unet_pipev7_single_attr_RGBs_smallBSZ_4gpus_bsz12_accumulate2-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-train_unet-loss_render1.0_lpips1.0-lr3e-05-Plat50/eval_global_step_5300_ckpt/model.safetensors
#     # --resume_unet /mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_unet/workspace_train/00009-train_unet_single_attr_RGBs_6gpus_bsz12_accumulate8-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-train_unet-loss_render1.0_lpips1.0-lr3e-05-Plat50/eval_global_step_2700_ckpt/model.safetensors


# # [MAY 27] inference on finetuning unet for single attribute image: OPACITY
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared_inference.py big \
#     --workspace runs/finetune_unet/workspace_debug \
#     --lr 1e-4 --num_epochs 10001 --eval_iter 20 --save_iter 20 --lr_scheduler Plat \
#     --lr_scheduler_patience 5 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'inference_unet_single_attr_OPACITY_pipev7_no_seq_NO_POS_EMBED_steps800_smallBSZ_seed42' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K_CLUSTER} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_CLUSTER} \
#     --set_random_seed --batch_size 1 --num_workers 2 \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v7_no_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --only_train_attention --rendering_loss_use_weight_t \
#     --inference_finetuned_unet --gradient_accumulation_steps 5 --output_size 320 \
#     --log_each_attribute_loss --train_unet_single_attr "opacity" --save_cond \
#     --resume_unet /mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_unet/workspace_train/00015-train_unet_pipev7_no_seq_NO_POS_EMBED_single_attr_OPACITYs_smallBSZ_4gpus_bsz12_accumulate1-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-train_unet-loss_render1.0_lpips1.0-lr3e-05-Plat50/eval_global_step_800_ckpt/model.safetensors


# [MAY 27] inference on finetuning unet for two attribute image with CD ATTN: RGBs + SCALE
accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared_inference.py big \
    --workspace runs/finetune_unet/workspace_inference \
    --lr 1e-4 --num_epochs 10001 --eval_iter 20 --save_iter 20 --lr_scheduler Plat \
    --lr_scheduler_patience 5 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
    --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
    --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
    --desc 'inference_unet_2attr_RGBs_SCALE_pipev7_CD_NO_POS_EMBED_steps1800_BSZ24_seed42' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K_CLUSTER} \
    --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_CLUSTER} \
    --set_random_seed --batch_size 1 --num_workers 2 \
    --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
    --scale_clamp_max -2 --scale_clamp_min -10 \
    --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
    --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
    --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
    --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
    --bg 1.0 --fovy 50 --only_train_attention --rendering_loss_use_weight_t \
    --inference_finetuned_unet --gradient_accumulation_steps 5 --output_size 320 \
    --log_each_attribute_loss --train_unet_single_attr "rgbs" "scale" \
    --resume_unet /mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_unet/workspace_train/00016-train_unet_pipev7_CD_attn_NO_POS_EMBED_2attr_RGBs_SCALE_smallBSZ_4gpus_bsz6_accumulate1-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-train_unet-loss_render1.0_lpips1.0-lr3e-05-Plat50/eval_global_step_1800_ckpt/model.safetensors