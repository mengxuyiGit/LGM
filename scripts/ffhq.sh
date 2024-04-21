DATA_DIR_DESK='/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/0d83a6b0d3dc4a3b8544fff507c04d86'
DATA_DIR_PINK_IRONMAN='/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/1dbcffe2f80b4d3ca50ff6406ab81f84'
DATA_DIR_HYDRANT='/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/0a9b36d36e904aee8b51e978a7c0acfd'
DATA_DIR_LAMP='/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/0c58250e3a7242e9bf21b114f2c8dce6'
DATA_DIR_BATCH="/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999"
DATA_DIR_BATCH_SUBSET="/home/xuyimeng/Data/lrm-zero123/assets/9000-9999-subset"

DATA_DIR_BATCH_SRN_CARS="/home/xuyimeng/Data/SRN/srn_cars/cars_train"

# DATA_DIR_BATCH_EG3D="/home/xuyimeng/Repo/eg3d/eg3d/out_lgm"
# DATA_DIR_BATCH_EG3D="/home/xuyimeng/Repo/eg3d/eg3d/out_lgm_5e-2"
DATA_DIR_BATCH_EG3D="/home/xuyimeng/Repo/eg3d/eg3d/out_lgm_3e-2"
# [APR 17] fit eg3d
accelerate launch --config_file acc_configs/gpu1.yaml main_pretrained_batch_ffhq.py big \
    --workspace runs/LGM_optimize_splatter/debug_eg3d \
    --resume pretrained/model_fp16.safetensors --num_epochs 2000 --fix_pretrained --prob_cam_jitter 0 \
    --lr 0.006 --num_input_views 6 --num_views 20 --use_adamW --lr_scheduler 'Plat' --lr_scheduler_patience 2 \
    --eval_iter 5 --save_iter 50 --desc 'batch' --data_path ${DATA_DIR_BATCH_EG3D} \
    --fovy 48.454 --output_size 64 --angle_y_step 3e-2
    # --fovy 18.837