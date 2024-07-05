# DATA_RENDERING_ROOT_LVIS_46K=/mnt/lingjie_cache/lvis_dataset/testing
DATA_RENDERING_ROOT_LVIS_46K_CLUSTER=/home/chenwang/data/lvis_dataset/testing

# debug training
    # 
export CUDA_VISIBLE_DEVICES=6
accelerate launch --config_file acc_configs/gpu1.yaml main1_lvis.py big --workspace runs/finetune_lgm/workspace_debug \
    --resume pretrained/fov60_lvis_model_8epochs.safetensors \
    --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K_CLUSTER} \
    --prob_cam_jitter 0 \
    --num_input_views 6 --fovy 60 --output_size 320 \
    --batch_size 1 --gradient_accumulation_steps 1

# export CUDA_VISIBLE_DEVICES=0
# accelerate launch --config_file acc_configs/gpu1.yaml main1_lvis.py big --workspace runs/finetune_lgm/workspace_debug \
#     --resume /mnt/kostas_home/lilym/LGM/LGM/pretrained/model_fp16.safetensors \
#     --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K_CLUSTER} \
#     --prob_cam_jitter 0 \
#     --num_input_views 6 --fovy 50 --output_size 320 \
#     --batch_size 4 --gradient_accumulation_steps 1

# # training (should use slurm)
# accelerate launch --config_file acc_configs/gpu8.yaml main.py big --workspace workspace
