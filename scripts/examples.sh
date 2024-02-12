# debug training: fix pretrained 
accelerate launch --config_file acc_configs/gpu1.yaml main_pretrained.py big --workspace workspace_debug_pretrained \
    --resume pretrained/model_fp16.safetensors --num_epochs 1000 --fix_pretrained \
    --lr 0.0002

# debug training
# accelerate launch --config_file acc_configs/gpu1.yaml main.py big --workspace workspace_ft --resume pretrained/model_fp16.safetensors --num_epochs 1000
# accelerate launch --config_file acc_configs/gpu1.yaml main.py big --workspace workspace_debug

# # training (should use slurm)
# accelerate launch --config_file acc_configs/gpu8.yaml main.py big --workspace workspace

# # test
# python infer.py big --workspace workspace_test --resume workspace/model.safetensors --test_path data_test

# # gradio app
# python app.py big --resume workspace/model.safetensors

# # local gui
# python gui.py big --output_size 800 --test_path workspace_test/anya_rgba.ply

# # mesh conversion
# python convert.py big --test_path workspace_test/anya_rgba.ply