
DATA_DIR_BATCH_RENDERING='.../assets/9000-9999'
DATA_DIR_BATCH_SPLATTER_GT_ROOT='.../workspace_debug_batch_subset_es'

### 2GPUs - with latents of size 16x16
accelerate launch --main_process_port 29510 --config_file acc_configs/gpu2.yaml main_zero123plus_v4_batch.py big --workspace runs/zerp123plus_batch/workspace_train \
    --lr 2e-4 --num_epochs 10001 --eval_iter 10 --save_iter 10 --lr_scheduler Plat --lr_scheduler_patience 5 \
    --prob_cam_jitter 0 --input_size 128 --num_input_views 6 --num_views 20 \
    --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 0 \
    --desc 'train-2gpu-latent16' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
    --set_random_seed --batch_size 10 --num_workers 6 --plot_attribute_histgram 'scale' \
    --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable


### 2GPUs - with latents of size 40x40. (Batchsize can be adjusted if have more memory)
accelerate launch --main_process_port 29511 --config_file acc_configs/gpu2.yaml main_zero123plus_v4_batch.py big --workspace runs/zerp123plus_batch/workspace_train \
    --lr 2e-4 --num_epochs 10001 --eval_iter 10 --save_iter 10 --lr_scheduler Plat --lr_scheduler_patience 5 \
    --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
    --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 0 \
    --desc 'train-2gpu-latent40' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
    --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
    --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable


### 1GPU - with latents of size 40x40.
accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch.py big --workspace runs/zerp123plus_batch/workspace_train \
    --lr 2e-4 --num_epochs 10001 --eval_iter 10 --save_iter 10 --lr_scheduler Plat --lr_scheduler_patience 5 \
    --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
    --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 0 \
    --desc 'train-latent40' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
    --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
    --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable