# DATA_RENDERING_ROOT_LVIS_46K=/mnt/lingjie_cache/lvis_dataset/testing
DATA_RENDERING_ROOT_LVIS_46K=/home/chenwang/data/lvis_dataset/testing

# debug training
    # 
# export CUDA_VISIBLE_DEVICES=3
accelerate launch --config_file acc_configs/gpu1.yaml main1_lvis.py big_320 --workspace runs/finetune_lgm/workspace_ovft \
    --resume pretrained/fov60_lvis_model_8epochs.safetensors \
    --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
    --prob_cam_jitter 0 \
    --num_input_views 6 --fovy 60 \
    --lambda_normal 0.005 \
    --batch_size 1 --gradient_accumulation_steps 1 --overfit_one_scene --desc "ablation_res320-normal_reg_0.005_after20k"

    # --resume runs/finetune_lgm/workspace_ovft/20240813-021423_ovft_1_resume_epoch10-add_normal0.2_dist0_loss/model.safetensors \
    # --resume runs/finetune_lgm/workspace_debug/20240813-011052_ovft_1/model.safetensors \
    # --resume runs/finetune_lgm/workspace_ovft/20240813-214830_big_320_out512-ovft_1_resume_epoch19_w_normal_reg-add_normal0.05_dist0_loss_after2k/iteration_2000/model.safetensors \
#  --output_size 320 
# TODO: ADD normal and depth loss
# also has ssim loss (2DGS)


    # --resume /mnt/kostas_home/lilym/LGM/LGM/runs/finetune_lgm/workspace_debug/20240812-200809/model.safetensors \
    # --resume pretrained/model_fp16_fixrot.safetensors \


# export CUDA_VISIBLE_DEVICES=0
# accelerate launch --config_file acc_configs/gpu1.yaml main1_lvis.py big --workspace runs/finetune_lgm/workspace_debug \
#     --resume /mnt/kostas_home/lilym/LGM/LGM/pretrained/model_fp16.safetensors \
#     --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K_CLUSTER} \
#     --prob_cam_jitter 0 \
#     --num_input_views 6 --fovy 50 --output_size 320 \
#     --batch_size 4 --gradient_accumulation_steps 1

# # training (should use slurm)
# accelerate launch --config_file acc_configs/gpu8.yaml main.py big --workspace workspace
