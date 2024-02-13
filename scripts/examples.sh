# debug training: fix pretrained 
#  accelerate launch --config_file acc_configs/gpu1.yaml main_cw.py big \
#     --workspace workspace_debug_pretrained_tf \
#     --resume pretrained/model_fp16.safetensors --num_epochs 100 --fix_pretrained \
#     --lr 0.0001 --num_views 6 --desc 'cw_dataloader' --eval_iter 1

accelerate launch --config_file acc_configs/gpu1.yaml main_pretrained.py big \
    --workspace workspace_debug_pretrained_tf \
    --resume pretrained/model_fp16.safetensors --num_epochs 1000 --fix_pretrained \
    --lr 0.0001 --num_input_views 6 --num_views 56 --desc 'fix_input_views' --eval_iter 10

# debug training
# accelerate launch --config_file acc_configs/gpu1.yaml main.py big --workspace workspace_ft --resume pretrained/model_fp16.safetensors --num_epochs 1000
# accelerate launch --config_file acc_configs/gpu1.yaml main.py big --workspace workspace_debug

# # training (should use slurm)
# accelerate launch --config_file acc_configs/gpu8.yaml main.py big --workspace workspace

# # test
# python infer_debug.py big --workspace workspace_test_debug_cam2 --resume pretrained/model_fp16.safetensors --test_path data_test
# python infer.py big --workspace workspace_test --resume workspace/model.safetensors --test_path data_test
# python infer_cw.py big --workspace workspace_test_debug_cam --resume pretrained/model_fp16.safetensors --test_path '/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/0a9b36d36e904aee8b51e978a7c0acfd' --num_input_views 6

# # gradio app
# python app.py big --resume workspace/model.safetensors

# # local gui
# python gui.py big --output_size 800 --test_path workspace_test/anya_rgba.ply

# # mesh conversion
# python convert.py big --test_path workspace_test/anya_rgba.ply