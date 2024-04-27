DATA_DIR_DESK='/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/0d83a6b0d3dc4a3b8544fff507c04d86'
DATA_DIR_PINK_IRONMAN='/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/1dbcffe2f80b4d3ca50ff6406ab81f84'
DATA_DIR_HYDRANT='/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/0a9b36d36e904aee8b51e978a7c0acfd'
DATA_DIR_LAMP='/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/0c58250e3a7242e9bf21b114f2c8dce6'
DATA_DIR_BATCH="/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999"
DATA_DIR_BATCH_SUBSET="/home/xuyimeng/Data/lrm-zero123/assets/9000-9999-subset"

DATA_DIR_BATCH_SRN_CARS="/home/xuyimeng/Data/SRN/srn_cars/cars_train"

# #### [FEB 19] Fast fitting - Batch process
# accelerate launch --config_file acc_configs/gpu1.yaml main_pretrained_batch.py big \
#     --workspace runs/LGM_optimize_splatter/inference \
#     --resume pretrained/model_fp16.safetensors --num_epochs 1 --fix_pretrained --prob_cam_jitter 0 \
#     --lr 0.003 --num_input_views 6 --num_views 20 --use_adamW --lr_scheduler 'Plat' --lr_scheduler_patience 2 \
#     --eval_iter 5 --save_iter 500 --desc 'batch' --data_path ${DATA_DIR_BATCH} \
#     --scene_start_index 650 --scene_end_index 750 --early_stopping \
#     --resume_workspace 'runs/LGM_optimize_splatter/workspace_debug_batch_subset_es/00006-batch-es10-Plat-patience_2-factor_0.5-eval_5-adamW-subset_650_750_splat128-inV6-lossV20-lr0.003'

# # #### [MAR 05] Fast fitting - Batch process
# # export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_pretrained_batch_srn_cars.py big --workspace runs/LGM_optimize_splatter/debug_resume \
#     --desc 'debug_srn_cars' --data_path ${DATA_DIR_BATCH_SRN_CARS} \
#     --resume pretrained/model_fp16.safetensors --num_epochs 300 --fix_pretrained --prob_cam_jitter 0 \
#     --lr 0.006 --num_input_views 6 --num_views 20 --use_adamW --lr_scheduler 'Plat' --lr_scheduler_patience 10 \
#     --eval_iter 150 --save_iter 500 --fovy 52 \
#     --scene_start_index 0 --scene_end_index 250 --early_stopping \
#     --resume_workspace '/home/xuyimeng/Repo/LGM/runs/LGM_optimize_splatter/shapenet_fit_batch/00000-debug_srn_cars-es10-Plat-patience_10-factor_0.5-eval_150-adamW-subset_0_250_splat128-inV6-lossV20-lr0.006/cars_train'


# resume from es checkpoint
# export CUDA_VISIBLE_DEVICES=1
accelerate launch --config_file acc_configs/gpu1.yaml main_pretrained_batch_srn_cars.py big --workspace runs/LGM_optimize_splatter/optimize \
    --desc 'further_optimize_splatter_srn_cars' --data_path ${DATA_DIR_BATCH_SRN_CARS} \
    --resume pretrained/model_fp16.safetensors --num_epochs 1000 --fix_pretrained --prob_cam_jitter 0 \
    --lr 0.006 --num_input_views 6 --num_views 20 --use_adamW --lr_scheduler 'Plat' --lr_scheduler_patience 10 \
    --eval_iter 20 --save_iter 500 --fovy 52 \
    --scene_start_index 500 --scene_end_index 750 --early_stopping \
    --further_optimize_splatter \
    --resume_workspace '/home/xuyimeng/Repo/LGM/runs/LGM_optimize_splatter/shapenet_fit_batch/00002-debug_srn_cars-es10-Plat-patience_10-factor_0.5-eval_150-adamW-subset_500_750_splat128-inV6-lossV20-lr0.006/cars_train' \
    --workspace_to_save "LGM/runs/LGM_optimize_splatter/optimize/00002-further_optimize_splatter_srn_cars-es10-Plat-patience_10-factor_0.5-eval_20-adamW-subset_500_750_splat128-inV6-lossV20-lr0.006"
