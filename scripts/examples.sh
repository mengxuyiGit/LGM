DATA_DIR_DESK='/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/0d83a6b0d3dc4a3b8544fff507c04d86'
DATA_DIR_PINK_IRONMAN='/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/1dbcffe2f80b4d3ca50ff6406ab81f84'
DATA_DIR_HYDRANT='/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/0a9b36d36e904aee8b51e978a7c0acfd'
DATA_DIR_LAMP='/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/0c58250e3a7242e9bf21b114f2c8dce6'

# debug training: fix pretrained 
#  accelerate launch --config_file acc_configs/gpu1.yaml main_cw.py big \
#     --workspace workspace_debug_pretrained_tf \
#     --resume pretrained/model_fp16.safetensors --num_epochs 100 --fix_pretrained \
#     --lr 0.0001 --num_views 6 --desc 'cw_dataloader' --eval_iter 1

# accelerate launch --config_file acc_configs/gpu1.yaml main_pretrained.py big \
#     --workspace runs/workspace_debug_pretrained_tf \
#     --resume pretrained/model_fp16.safetensors --num_epochs 10001 --fix_pretrained \
#     --lr 0.0006 --num_input_views 6 --num_views 20 --desc 'desk' --eval_iter 100 \
#     --prob_cam_jitter 0 --data_path '/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/0d83a6b0d3dc4a3b8544fff507c04d86'
    
accelerate launch --config_file acc_configs/gpu1.yaml main_pretrained.py big \
    --workspace runs/LGM_optimize_splatter/workspace_splatter_gt_full_ply_fixed_einops \
    --resume pretrained/model_fp16.safetensors --num_epochs 10001 --fix_pretrained \
    --lr 0.0006 --num_input_views 6 --num_views 20 --desc 'hydrant-adamW' --eval_iter 200 \
    --prob_cam_jitter 0 --data_path ${DATA_DIR_HYDRANT}

# accelerate launch --config_file acc_configs/gpu1.yaml main_pretrained.py big \
#     --workspace runs/LGM_optimize_splatter/workspace_debug \
#     --resume pretrained/model_fp16.safetensors --num_epochs 10001 --fix_pretrained \
#     --lr 0.0006 --num_input_views 6 --num_views 20 --desc 'hydrant-gt' --eval_iter 2 \
#     --prob_cam_jitter 0 --data_path ${DATA_DIR_HYDRANT} --eval_splatter_gt
    
    # pink_ironman: '/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/1dbcffe2f80b4d3ca50ff6406ab81f84'

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