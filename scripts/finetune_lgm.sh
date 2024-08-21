# DATA_RENDERING_ROOT_LVIS_46K=/mnt/lingjie_cache/lvis_dataset/testing
DATA_RENDERING_ROOT_LVIS_46K=/home/chenwang/data/lvis_dataset/testing
LARA_h5=/home/xuyimeng/Repo/LaRa/dataset/gobjaverse/gobjaverse.h5

 CUDA_VISIBLE_DEVICES=0
accelerate launch --config_file acc_configs/gpu1.yaml main1_lvis.py big --workspace runs/finetune_lgm/workspace_ovft \
    --data_path_rendering ${LARA_h5} --data_mode lara --fovy 39.6 --input_size 256 \
    --resume /mnt/kostas_home/lilym/LGM/LGM/runs/finetune_lgm/workspace_ovft/20240820-010207_detach-splatter_128-normal_loss_0.2_depth_0.5_after500-normal_err_after7k/model_epoch_6/model.safetensors \
    --prob_cam_jitter 0 \
    --num_input_views 6 \
    --lambda_normal_err 0.0 --lambda_normal 0.2 --lambda_depth 1.0 --normal_depth_begin_iter 500 --resume_iter 7000 \
    --batch_size 1 --gradient_accumulation_steps 1 --overfit_one_scene --desc "detach-resume_epoch6-LARGER_Depth_loss-No_normal_err-splatter_128-normal_loss_0.2_depth_1.0_after500"


# export CUDA_VISIBLE_DEVICES=0
# accelerate launch --config_file acc_configs/gpu1.yaml main1_lvis.py big --workspace runs/finetune_lgm/workspace_ovft \
#     --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} --data_mode gbuffer --fovy 39.6 \
#     --resume /mnt/kostas_home/lilym/LGM/LGM/runs/finetune_lgm/workspace_ovft/20240820-010207_detach-splatter_128-normal_loss_0.2_depth_0.5_after500-normal_err_after7k/model_epoch_6/model.safetensors \
#     --prob_cam_jitter 0 \
#     --num_input_views 6 \
#     --lambda_normal_err 0.0 --lambda_normal 0.2 --lambda_depth 1.0 --normal_depth_begin_iter 500 --resume_iter 7000 \
#     --batch_size 1 --gradient_accumulation_steps 1 --overfit_one_scene --desc "detach-resume_epoch6-LARGER_Depth_loss-No_normal_err-splatter_128-normal_loss_0.2_depth_1.0_after500"

# #    --data_mode gbuffer  
#     # --resume pretrained/model_fp16_fixrot.safetensors \
#     # --resume /mnt/kostas_home/lilym/LGM/LGM/runs/finetune_lgm/workspace_ovft/20240819-200933_load_gobjaverse-normal_loss_0.2-depth_0.5-after_2k-no_normal_err/model_epoch_2/model.safetensors \
#     # --resume pretrained/fov60_lvis_model_8epochs.safetensors \
#     # --resume runs/finetune_lgm/workspace_ovft/20240813-021423_ovft_1_resume_epoch10-add_normal0.2_dist0_loss/model.safetensors \
#     # --resume runs/finetune_lgm/workspace_debug/20240813-011052_ovft_1/model.safetensors \
#     # --resume runs/finetune_lgm/workspace_ovft/20240813-214830_big_320_out512-ovft_1_resume_epoch19_w_normal_reg-add_normal0.05_dist0_loss_after2k/iteration_2000/model.safetensors \
# #  --output_size 320 
# # TODO: ADD normal and depth loss
# # also has ssim loss (2DGS)


