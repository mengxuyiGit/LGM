DATA_RENDERING_ROOT_LVIS_46K=/mnt/lingjie_cache/lvis_dataset/testing
DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT=/mnt/kostas-graid/datasets/xuyimeng/lvis/data_processing_finetuned_lgm_fov60_8epochs/testing
# DATA_RENDERING_ROOT_LVIS_46K_CLUSTER=/home/chenwang/data/lvis_dataset/testing
# DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_CLUSTER=/mnt/kostas-graid/datasets/xuyimeng/lvis/data_processing_finetuned_lgm_fov60_8epochs/testing

# export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch --main_process_port 29518 --config_file acc_configs/gpu8.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
    --workspace runs/finetune_unet/workspace_train_july \
    --lr 4e-6 --max_train_steps 100000 --eval_iter 500 --save_iter 500 --lr_scheduler Plat \
    --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
    --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
    --lambda_splatter 0 --lambda_rendering 0 --lambda_alpha 0 --lambda_lpips 0 --lambda_splatter_lpips 0 \
    --desc 'train_unet-fted_LGM_fov60' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
    --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
    --set_random_seed \
    --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
    --custom_pipeline "./zero123plus/pipeline_v8_cat.py" --render_input_views --attr_group_mode "v5" \
    --bg 1.0 --fovy 60 --rendering_loss_use_weight_t \
    --train_unet --class_emb_cat  --drop_cond_prob 0.1 --only_train_attention \
    --batch_size 1 --num_workers 1 --gradient_accumulation_steps 4 \
    --invalid_list /mnt/kostas_home/lilym/LGM/LGM/data_lists/lvis_invalid_uids_nineviews.json \
    --use_video_decoderST --resume_decoder /mnt/kostas_home/lilym/LGM/LGM/runs/finetune_decoder/workspace_train_july/20240708-050809-svd-decoder-fted_lgm-resume6k-loss_render1.0_splatter2.0_lpips2.0-lr0.0001-Plat5/eval_global_step_30000_ckpt/model.safetensors

# [before finetune LGM]
    # --resume_decoder /mnt/kostas_home/lilym/LGM/LGM/runs/finetune_decoder/workspace_train_june/00000-sd_decoder-bsz32-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render0.1_splatter1.0_lpips0.1-lr2e-06-Plat5/eval_global_step_27000_ckpt/model.safetensors \
    # --resume_unet /mnt/kostas_home/lilym/LGM/LGM/runs/finetune_unet/workspace_train_june/00019-train_unet-exp_act-with_timeproj_clsemb-resume29k-bsz64-NO-rendering-only_attn-pipev8-resum50kckpt-lr1e-6-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-train_unet-loss-lr4e-06-Plat50/eval_global_step_20000_ckpt/model.safetensors \
   