DATA_RENDERING_ROOT_LVIS_46K=/mnt/lingjie_cache/lvis_dataset/testing
DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT=/mnt/lingjie_cache/lvis_splatters/testing

# [July 01] refine_net: ptv3
export CUDA_VISIBLE_DEVICES=1
export SP_CONV_VERBOSE=1
export CUDA_LAUNCH_BLOCKING=1
export SP_CONV_TUNING=0
accelerate launch --main_process_port 29516 --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_unet_accumulate_shared.py big \
    --workspace runs/finetune_refinenet/workspace_overfit \
    --lr 2e-4 --max_train_steps 100000 --eval_iter 100 --save_iter 1000 --lr_scheduler Plat \
    --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
    --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
    --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 10 --lambda_splatter_lpips 0 \
    --desc 'refinement-overfit_unet_inference-input_3choices-batch_denoise' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
    --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
    --set_random_seed \
    --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
    --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
    --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
    --custom_pipeline "./zero123plus/pipeline_v8_cat.py" --render_input_views --attr_group_mode "v5" \
    --bg 1.0 --fovy 50 --rendering_loss_use_weight_t \
    --train_refine_net --class_emb_cat \
    --use_video_decoderST --resume_decoder /mnt/kostas_home/lilym/LGM/LGM/runs/finetune_decoder/workspace_train_june/00002-svd_decoder-bsz32-resume19.5k-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render0.1_splatter1.0_lpips0.1-lr1e-06-Plat5/eval_global_step_27500_ckpt/model.safetensors \
    --resume_unet /mnt/kostas_home/lilym/LGM/LGM/runs/finetune_unet/workspace_train_june/00024-train_unet-RENDERING_LOSS-resume00019_timeproj20k_sd_decoder-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-train_unet-loss_render10.0_lpips10.0-lr4e-06-Plat50/eval_global_step_99500_ckpt/model.safetensors \
    --invalid_list /mnt/kostas_home/lilym/LGM/LGM/data_lists/lvis_invalid_uids_nineviews.json \
    --drop_cond_prob 0.1 --refinement_network ptv3 --guidance_scale 2.0 \
    --batch_size 4 --num_workers 1 --gradient_accumulation_steps 1 --overfit_one_scene


    # --resume_decoder /mnt/kostas_home/lilym/LGM/LGM/runs/finetune_decoder/workspace_train_june/00000-sd_decoder-bsz32-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render0.1_splatter1.0_lpips0.1-lr2e-06-Plat5/eval_global_step_27000_ckpt/model.safetensors \
    # --resume_unet /mnt/kostas_home/lilym/LGM/LGM/runs/finetune_unet/workspace_train_june/00019-train_unet-exp_act-with_timeproj_clsemb-resume29k-bsz64-NO-rendering-only_attn-pipev8-resum50kckpt-lr1e-6-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-train_unet-loss-lr4e-06-Plat50/eval_global_step_20000_ckpt/model.safetensors \
