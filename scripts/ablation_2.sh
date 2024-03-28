
DATA_DIR_BATCH_RENDERING='/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999'
DATA_DIR_BATCH_SPLATTER_GT_ROOT='/home/xuyimeng/Repo/LGM/runs/LGM_optimize_splatter/workspace_debug_batch_subset_es'

accelerate launch --config_file acc_configs/gpu4.yaml main_zero123plus_v4_batch_code.py big --workspace runs/zerp123plus_batch/workspace_ablation2 \
    --lr 2e-4 --num_epochs 20001 --eval_iter 50 --save_iter 50 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
    --prob_cam_jitter 0 --prob_grid_distortion 0 --input_size 320 --num_input_views 6 --num_views 20 \
    --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 0 \
    --desc 'ablation_2_only_decoder_4gpus' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
    --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
    --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
    --model_type Zero123PlusGaussianCode \
    --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
    --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder