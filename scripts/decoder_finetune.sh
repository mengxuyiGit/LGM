DATA_RENDERING_ROOT_LVIS_46K=/mnt/lingjie_cache/lvis_dataset/testing
DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT=/mnt/lingjie_cache/lvis_splatters/testing

export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch --main_process_port 29514 --config_file acc_configs/gpu4.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_accumulate.py big \
    --workspace runs/finetune_decoder/workspace_train \
    --lr 1e-4 --max_train_steps 30000 --eval_iter 100 --save_iter 100 --lr_scheduler Plat \
    --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
    --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
    --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
    --desc '4gpus_bsz2_accumulate32' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
    --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
    --set_random_seed --batch_size 2 --num_workers 1 \
    --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
    --scale_clamp_max -2 --scale_clamp_min -10 \
    --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
    --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
    --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
    --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
    --bg 1.0 --fovy 50 --only_train_attention --rendering_loss_use_weight_t \
    --finetune_decoder --gradient_accumulation_steps 32 --output_size 320

# # singleGPU
# # export CUDA_VISIBLE_DEVICES=5
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_accumulate.py big \
#     --workspace runs/finetune_decoder/workspace_debug \
#     --lr 1e-4 --num_epochs 10001 --eval_iter 10 --save_iter 10 --lr_scheduler Plat \
#     --lr_scheduler_patience 5 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'debug_global_step_tensorboard' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
#     --set_random_seed --batch_size 2 --num_workers 2 \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v7_seq.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 50 --only_train_attention --rendering_loss_use_weight_t \
#     --finetune_decoder --gradient_accumulation_steps 5 --output_size 320 
#     # --skip_training
