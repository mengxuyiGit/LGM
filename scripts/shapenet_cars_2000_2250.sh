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

# #### [MAR 05] Fast fitting - Batch process
accelerate launch --config_file acc_configs/gpu1.yaml main_pretrained_batch_srn_cars.py big --workspace runs/LGM_optimize_splatter/shapenet_fit_batch \
    --desc 'debug_srn_cars' --data_path ${DATA_DIR_BATCH_SRN_CARS} \
    --resume pretrained/model_fp16.safetensors --num_epochs 300 --fix_pretrained --prob_cam_jitter 0 \
    --lr 0.006 --num_input_views 6 --num_views 20 --use_adamW --lr_scheduler 'Plat' --lr_scheduler_patience 10 \
    --eval_iter 150 --save_iter 500 --fovy 52 \
    --scene_start_index 2000 --scene_end_index 2250 --early_stopping \
    --resume_workspace 'runs/LGM_optimize_splatter/shapenet_fit_batch/00008-debug_srn_cars-es10-Plat-patience_10-factor_0.5-eval_150-adamW-subset_2000_2250_splat128-inV6-lossV20-lr0.006'


# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_pretrained_batch_srn_cars2.py big --workspace runs/LGM_optimize_splatter/shapenet_test_time \
#     --desc 'debug_srn_cars' --data_path ${DATA_DIR_BATCH_SRN_CARS} \
#     --resume pretrained/model_fp16.safetensors --num_epochs 300 --fix_pretrained --prob_cam_jitter 0 \
#     --lr 0.01 --num_input_views 6 --num_views 20 --use_adamW --lr_scheduler 'Plat' --lr_scheduler_patience 10 \
#     --eval_iter 50 --save_iter 500 --fovy 52 \
#     --scene_start_index 0 --scene_end_index 4 --early_stopping \
#     --resume_workspace "runs/LGM_optimize_splatter/shapenet_test_time/00002-debug_srn_cars-es10-Plat-patience_10-factor_0.5-eval_50-adamW-subset_0_4_splat128-inV6-lossV20-lr0.01"
#     # --znear 0.8 --zfar 1.8

# --scene_start_index 750 --scene_end_index 1000 --early_stopping
# --scene_start_index 1250 --scene_end_index 1500 --early_stopping

# #### [MAR 29] Change the splatter rep to be depth and offset. Fast fitting - Batch process
# accelerate launch --config_file acc_configs/gpu1.yaml main_pretrained_batch.py big \
#     --workspace runs/LGM_optimize_splatter/debug \
#     --resume pretrained/model_fp16.safetensors --num_epochs 1 --fix_pretrained --prob_cam_jitter 0 \
#     --lr 0.003 --num_input_views 6 --num_views 20 --use_adamW --lr_scheduler 'Plat' --lr_scheduler_patience 2 \
#     --eval_iter 5 --save_iter 500 --desc 'depth_offset' --data_path ${DATA_DIR_BATCH} \
#     --scene_start_index 0 --scene_end_index -1 --early_stopping \
#     --use_splatter_with_de    pth_offset


# #### [MAR 29] Change the splatter rep to be depth and offset. Fast fitting - Batch process
# accelerate launch --config_file acc_configs/gpu1.yaml main_pretrained_batch.py big --workspace runs/LGM_optimize_splatter/debug2 \
#     --desc 'apr-6-fix-depth-render' \
#     --resume pretrained/model_fp16.safetensors --num_epochs 2000 --fix_pretrained --prob_cam_jitter 0 \
#     --lr 0.006 --num_input_views 6 --num_views 20 --use_adamW --lr_scheduler 'Plat' --lr_scheduler_patience 20 \
#     --eval_iter 5 --save_iter 200 --desc 'depth_offset' --data_path ${DATA_DIR_BATCH} \
#     --scene_start_index 0 --scene_end_index -1 --early_stopping \
#     --use_splatter_with_depth_offset --always_zero_xy_offset --save_raw_tensor_splatter
#     # --zero_init_xy_offset


# #### [FEB 19] Fast fitting - Batch process - compare: bad optimizer
# accelerate launch --config_file acc_configs/gpu1.yaml main_pretrained_batch.py big \
#     --workspace runs/LGM_optimize_splatter/workspace_debug \
#     --resume pretrained/model_fp16.safetensors --num_epochs 2001 --fix_pretrained --prob_cam_jitter 0 \
#     --lr 0.0006 --num_input_views 6 --num_views 20 --use_adamW --lr_scheduler 'CosAnn' --lr_scheduler_patience 2 \
#     --eval_iter 50 --save_iter 500 --desc 'batch' --data_path ${DATA_DIR_BATCH} --scene_start_index 2 --scene_end_index 10 

# debug training
# accelerate launch --config_file acc_configs/gpu1.yaml main.py big --workspace workspace_ft --resume pretrained/model_fp16.safetensors --num_epochs 1000
# accelerate launch --config_file acc_configs/gpu1.yaml main.py big --workspace workspace_debug

# # training (should use slurm)
# accelerate launch --config_file acc_configs/gpu8.yaml main.py big --workspace workspace

# # test
# python infer_debug.py big --workspace workspace_test_debug_cam3 --resume pretrained/model_fp16.safetensors --test_path data_test --num_input_views 6 --prob_cam_jitter 0
# python infer.py big --workspace workspace_test --resume workspace/model.safetensors --test_path data_test
# python infer_cw.py big --workspace workspace_test_debug_cam --resume pretrained/model_fp16.safetensors --test_path '/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/0a9b36d36e904aee8b51e978a7c0acfd' --num_input_views 6

# # gradio app
# python app.py big --resume workspace/model.safetensors

# # local gui
# python gui.py big --output_size 800 --test_path workspace_test/anya_rgba.ply

# # mesh conversion
# python convert.py big --test_path workspace_test/anya_rgba.ply