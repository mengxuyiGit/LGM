DATA_RENDERING_ROOT_LVIS_46K=/mnt/lingjie_cache/lvis_dataset/testing
# DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT=/mnt/lingjie_cache/lvis_splatters/testing
# DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_CLUSTER=/mnt/kostas-graid/datasets/xuyimeng/lvis/data_processing/testing
# DATA_RENDERING_ROOT_LVIS_46K_CLUSTER=/home/chenwang/data/lvis_dataset/testing
DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_FINETUNED_CLUSTER=/mnt/kostas-graid/datasets/xuyimeng/lvis/data_processing_finetuned_lgm_fov60_8epochs/testing

# [Jul 3] SD-decoder with finetuned LGM results
# export CUDA_VISIBLE_DEVICES=6,7
accelerate launch --main_process_port 29514 --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_accumulate.py big \
    --workspace runs/finetune_decoder/workspace_debug \
    --lr 1e-4 --num_epochs 10001 --eval_iter 500 --save_iter 500 --lr_scheduler Plat \
    --lr_scheduler_patience 5 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
    --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
    --lambda_splatter 2 --lambda_rendering 1 --lambda_alpha 1 --lambda_lpips 2 \
    --desc 'debug_local_dir-sd-decoder-fted_lgm' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
    --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_FINETUNED_CLUSTER} \
    --set_random_seed \
    --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
    --custom_pipeline "./zero123plus/pipeline_v8_cat.py" --render_input_views --attr_group_mode "v5" \
    --bg 1.0 --fovy 60 \
    --finetune_decoder \
    --batch_size 2 --num_workers 1 --gradient_accumulation_steps 1 \
    --invalid_list /mnt/kostas_home/lilym/LGM/LGM/data_lists/lvis_invalid_uids_nineviews.json