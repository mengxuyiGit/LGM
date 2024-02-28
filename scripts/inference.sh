DATA_DIR_BATCH_RENDERING='/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999'
DATA_DIR_BATCH_SPLATTER_GT_ROOT='/home/xuyimeng/Repo/LGM/runs/LGM_optimize_splatter/workspace_debug_batch_subset_es'

python zero123plus/img_to_splatter.py big \
    --lr 2e-4 --num_epochs 10001 --eval_iter 10 --save_iter 10 --lr_scheduler Plat --lr_scheduler_patience 5 \
    --prob_cam_jitter 0 --input_size 128 --num_input_views 6 --num_views 20 \
    --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 0 \
    --desc 'debug-inference' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
    --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
    --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
    --scale_clamp_max -2 --scale_clamp_min -10 --skip_training --inference_noise_level 75 \
    --resume "runs/zerp123plus_batch/workspace_train/20240226-012340-train-resume185634-920-aligned-sf-2gpu-v0_unfreeze_all-skip_predict_x0-loss_render1.0_splatter1.0-lr0.0002-Plat5/model.safetensors"