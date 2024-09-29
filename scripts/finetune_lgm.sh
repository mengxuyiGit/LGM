# DATA_RENDERING_ROOT_LVIS_46K=/mnt/lingjie_cache/lvis_dataset/testing
DATA_RENDERING_ROOT_LVIS_46K=/home/chenwang/data/lvis_dataset/testing
# LARA_h5=/home/xuyimeng/Repo/LaRa/dataset/gobjaverse/gobjaverse.h5
# LARA_h5=/home/xuyimeng/Repo/LaRa/outputs/gobjverse_hdf5_v1_00.hdf5
# LARA_h5=/home/xuyimeng/Repo/LaRa/outputs/gobjverse_hdf5_v2_raw_normal_00.hdf5
# LARA_h5=/mnt/kostas-graid/datasets/xuyimeng/GobjLara/dataset/gobjaverse/gobjaverse.h5
LARA_h5=/mnt/kostas-graid/datasets/xuyimeng/GobjLara_Sep21/dataset/gobjaverse/gobjaverse.h5

# # CUDA_VISIBLE_DEVICES=1,2,3,4
# CUDA_VISIBLE_DEVICES=0
# accelerate launch --config_file acc_configs/gpu1.yaml main1_lvis.py big --workspace runs/finetune_lgm/workspace_debug \
#     --resume runs/finetune_lgm/workspace_train_aug/00000_lara_h5_full-normal0.2_depth0.5_loss_0.5_after5000-no_normal_err/model_epoch_2/model.safetensors \
#     --data_path_rendering ${LARA_h5} --data_mode lara --fovy 39.6 --input_size 256 --num_views 10 \
#     --prob_cam_jitter 0 \
#     --num_input_views 6 \
#     --lambda_normal_err 0.0 --lambda_normal 0.2 --lambda_depth 0.5 --normal_depth_begin_iter 5000 --resume_iter 0 \
#     --batch_size 4 --gradient_accumulation_steps 1 --desc "white_bg_normal-resume_epoch2-view0_24-lara_h5_full-normal0.2_depth0.5_loss_0.5_after5000-no_normal_err" --overfit_one_scene

# CUDA_VISIBLE_DEVICES=1,2,3,4
# CUDA_VISIBLE_DEVICES=0
accelerate launch --config_file acc_configs/gpu8.yaml main1_lvis.py big --workspace runs/finetune_lgm/workspace_train_aug \
    --resume runs/finetune_lgm/workspace_train_aug/00002_resume_epoch2-view0_24-lara_h5_full-normal0.2_depth0.5_loss_0.5_after5000-no_normal_err/model_epoch_0_iter_6000/model.safetensors \
    --data_path_rendering ${LARA_h5} --data_mode lara --fovy 39.6 --input_size 256 --num_views 10 \
    --prob_cam_jitter 0 \
    --num_input_views 6 \
    --lambda_normal_err 0.0 --lambda_normal 0.2 --lambda_depth 0.5 --normal_depth_begin_iter 5000 --resume_iter 0 \
    --batch_size 4 --gradient_accumulation_steps 1 --desc "resume_lara00002"
    #  --overfit_one_scene


# # # [LVIS + 2DGS] Sep 21 
# # export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
# # accelerate launch --config_file acc_configs/gpu6.yaml main1_lvis.py big --workspace runs/finetune_lgm/workspace_train_sep \
# #     --resume pretrained/model_fp16_fixrot.safetensors \
# #     --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} --data_mode s3 --fovy 60 \
# #     --prob_cam_jitter 0 \
# #     --num_input_views 6 \
# #     --lambda_normal_err 0.0 --lambda_normal 0.2 --lambda_depth 1.0 --normal_depth_begin_iter 500 --resume_iter 0 \
# #     --batch_size 1 --gradient_accumulation_steps 1 --desc "lvis_2dgs"
# #     #  --overfit_one_scene 


# # [LVIS + 2DGS] Sep 22: rsume with large normal err
# export CUDA_VISIBLE_DEVICES=0,1,6,7
# accelerate launch --config_file acc_configs/gpu4.yaml main1_lvis.py big --workspace runs/finetune_lgm/workspace_train_sep \
#     --resume /mnt/kostas_home/lilym/LGM/LGM/runs/finetune_lgm/workspace_train_sep_wrong_fov/00002_lvis_2dgs-resume_larger_normal_err/model_epoch_5_iter_3000/model.safetensors \
#     --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} --data_mode s3 --fovy 60 \
#     --prob_cam_jitter 0 \
#     --num_input_views 6 \
#     --lambda_normal_err 0.0 --lambda_normal 0.3 --lambda_depth 1.0 --normal_depth_begin_iter 500 --resume_iter 0 \
#     --batch_size 2 --gradient_accumulation_steps 1 --desc "lvis_2dgs"

    
    # --resume /mnt/kostas_home/lilym/LGM/LGM/runs/finetune_lgm/workspace_train_aug/00002_resume_epoch2-view0_24-lara_h5_full-normal0.2_depth0.5_loss_0.5_after5000-no_normal_err/model_epoch_0_iter_6000/model.safetensors \
# # TODO: ADD normal and depth loss
# # also has ssim loss (2DGS)
