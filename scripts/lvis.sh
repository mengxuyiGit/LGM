DATA_DIR_BATCH_RENDERING='/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999'
DATA_DIR_BATCH_SPLATTER_GT_ROOT='/home/xuyimeng/Repo/LGM/runs/splatter_gt'
# DATA_DIR_BATCH_RENDERING='/mnt/kostas-graid/sw/envs/xuyimeng/data/testing/1000-1999'


accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference_marigold_v5_fake_init_optimize_splatter_lvis.py big \
    --workspace runs/lvis/workspace_debug \
    --resume pretrained/model_fp16_fixrot.safetensors \
    --lr 6e-3 --num_epochs 500 --eval_iter 5 --save_iter 5 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
    --prob_cam_jitter 0 --num_input_views 6 --num_views 20 --render_input_views \
    --bg 1.0 --fovy  47.1 \
    --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
    --desc 'lvis_optimize_splatter' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} \
    --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
    --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
    --save_train_pred -1 --decode_splatter_to_128 \
    --optimization_objective "splatter_images" --attr_group_mode "v5" \
    --rendering_loss_on_splatter_to_encode 
    
    # --output_size 320 --input_size 320 \
    # --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} 


# TODO: change --num_epochs 301 to larger
# 

# ### [APR 26 - Use fixed rotation] Fast fitting - Batch process
# accelerate launch --config_file acc_configs/gpu1.yaml main_pretrained_batch_no_param.py big \
#     --workspace runs/lvis/debug \
#     --resume pretrained/model_fp16_fixrot.safetensors --num_epochs 1 --fix_pretrained --prob_cam_jitter 0 \
#     --lr 0.003 --num_input_views 6 --num_views 20 --use_adamW --lr_scheduler 'Plat' --lr_scheduler_patience 2 \
#     --eval_iter 5 --save_iter 500 --desc 'batch' --data_path ${DATA_DIR_BATCH_RENDERING} \
#     --scene_start_index 0 --scene_end_index 5 --early_stopping