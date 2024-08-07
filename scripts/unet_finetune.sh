DATA_RENDERING_ROOT_LVIS_46K=/mnt/lingjie_cache/lvis_dataset/testing
DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT=/mnt/kostas-graid/datasets/xuyimeng/lvis/data_processing_finetuned_lgm_fov60_8epochs/testing
# DATA_RENDERING_ROOT_LVIS_46K=/home/chenwang/data/lvis_dataset/testing
# DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT=/mnt/kostas-graid/datasets/xuyimeng/lvis/data_processing_finetuned_lgm_fov60_8epochs/testing

# export CUDA_VISIBLE_DEVICES=1,4,5,6
# export CUDA_VISIBLE_DEVICES=2
accelerate launch --main_process_port 29519 --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
    --workspace runs/finetune_unet/workspace_debug \
    --lr 4e-6 --max_train_steps 100000 --eval_iter 1000 --save_iter 1000 --lr_scheduler Plat \
    --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
    --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 10 \
    --lambda_splatter 0 --lambda_rendering 0 --lambda_alpha 0 --lambda_lpips 0 --lambda_splatter_lpips 0 \
    --desc 'check_psnr-V9_expert_branch-opt_exert_resnet-NO_stats_norm-NO_cam_ray_emb' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
    --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
    --set_random_seed \
    --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
    --custom_pipeline "./zero123plus/pipeline_v9_expbranch.py" --render_input_views --attr_group_mode "v5" \
    --bg 1.0 --fovy 60 --rendering_loss_use_weight_t \
    --train_unet --class_emb_cat  --drop_cond_prob 0.1 --only_train_attention \
    --batch_size 2 --num_workers 1 --gradient_accumulation_steps 4 \
    --invalid_list /mnt/kostas_home/lilym/LGM/LGM/data_lists/lvis_invalid_uids_nineviews.json \
    --resume_decoder /mnt/kostas_home/lilym/LGM/LGM/runs/finetune_decoder/workspace_train_july/20240713-sd-decoder-fted_lgm-loss_render1.0_splatter2.0_lpips2.0-lr0.0001-Plat5/eval_global_step_1000_ckpt/model.safetensors
    
    # --latents_normalization_stats /mnt/kostas_home/lilym/LGM/LGM/runs/statistics/workspace_320real \
    # --diffusion_concat_camera_ray_embeddings \
    # TODO: when use pipeline_v9, can remove the temporal attn [Done]
    # TODO: make the inference scheduler pass 


# # [Aug 05] overfit
# export CUDA_VISIBLE_DEVICES=3
# accelerate launch --main_process_port 29517 --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
#     --workspace runs/finetune_unet/workspace_overfit_aug \
#     --lr 4e-6 --max_train_steps 100000 --eval_iter 200 --save_iter 1000 --lr_scheduler Plat \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 10 \
#     --lambda_splatter 0 --lambda_rendering 0 --lambda_alpha 0 --lambda_lpips 0 --lambda_splatter_lpips 0 \
#     --desc 'v9-opt_attentions-no_experts_resnet-NO_stats_norm-NO_cam_ray_emb-train_all_attr' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
#     --set_random_seed \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v9_expbranch.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 60 --rendering_loss_use_weight_t \
#     --train_unet --class_emb_cat  --drop_cond_prob 0.1 --only_train_attention \
#     --batch_size 2 --num_workers 1 --gradient_accumulation_steps 1 \
#     --invalid_list /mnt/kostas_home/lilym/LGM/LGM/data_lists/lvis_invalid_uids_nineviews.json \
#     --overfit_one_scene \
#     --resume_decoder /mnt/kostas_home/lilym/LGM/LGM/runs/finetune_decoder/workspace_train_july/20240713-sd-decoder-fted_lgm-loss_render1.0_splatter2.0_lpips2.0-lr0.0001-Plat5/eval_global_step_1000_ckpt/model.safetensors
#     # --latents_normalization_stats /mnt/kostas_home/lilym/LGM/LGM/runs/statistics/workspace_320real \
#     # --lambda_each_attribute_loss 0 0 1 0 \
#     # --diffusion_concat_camera_ray_embeddings \