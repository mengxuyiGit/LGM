# DATA_RENDERING_ROOT_LVIS_46K=/mnt/lingjie_cache/lvis_dataset/testing
DATA_RENDERING_ROOT_LVIS_46K=/home/chenwang/data/lvis_dataset/testing
# LARA_h5=/home/xuyimeng/Repo/LaRa/dataset/gobjaverse/gobjaverse.h5
# LARA_h5=/home/xuyimeng/Repo/LaRa/outputs/gobjverse_hdf5_v1_00.hdf5
# LARA_h5=/home/xuyimeng/Repo/LaRa/outputs/gobjverse_hdf5_v2_raw_normal_00.hdf5
LARA_h5=/mnt/kostas-graid/datasets/xuyimeng/GobjLara/dataset/gobjaverse/gobjaverse.h5

CUDA_VISIBLE_DEVICES=0
accelerate launch --config_file acc_configs/gpu1.yaml main1_lvis.py big --workspace runs/finetune_lgm/workspace_debug \
    --resume pretrained/model_fp16_fixrot.safetensors \
    --data_path_rendering ${LARA_h5} --data_mode lara --fovy 39.6 --input_size 256 --num_views 12 \
    --prob_cam_jitter 0 \
    --num_input_views 6 \
    --lambda_normal_err 0.0 --lambda_normal 0.2 --lambda_depth 0.5 --normal_depth_begin_iter 5000 --resume_iter 0 \
    --batch_size 2 --gradient_accumulation_steps 1 --desc "bsz2-lara_dataloader-No_normal_err-splatter_128-normal_loss_0.2_depth_1.0_after500"


# export CUDA_VISIBLE_DEVICES=0
# accelerate launch --config_file acc_configs/gpu1.yaml main1_lvis.py big --workspace runs/finetune_lgm/workspace_debug \
#     --resume pretrained/model_fp16_fixrot.safetensors \
#     --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} --data_mode gbuffer --fovy 39.6 \
#     --prob_cam_jitter 0 \
#     --num_input_views 6 \
#     --lambda_normal_err 0.0 --lambda_normal 0.2 --lambda_depth 1.0 --normal_depth_begin_iter 500 --resume_iter 7000 \
#     --batch_size 1 --gradient_accumulation_steps 1 --overfit_one_scene --desc "gbuffer-ROT-neg01_201_with_norm-resume_epoch6-LARGER_Depth_loss-No_normal_err-splatter_128-normal_loss_0.2_depth_1.0_after500"


# # TODO: ADD normal and depth loss
# # also has ssim loss (2DGS)
