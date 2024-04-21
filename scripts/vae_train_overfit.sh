DATA_DIR_DESK='/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/0d83a6b0d3dc4a3b8544fff507c04d86'
DATA_DIR_PINK_IRONMAN='/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/1dbcffe2f80b4d3ca50ff6406ab81f84'
DATA_DIR_HYDRANT='/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/0a9b36d36e904aee8b51e978a7c0acfd'
DATA_DIR_LAMP='/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999/0c58250e3a7242e9bf21b114f2c8dce6'

DATA_DIR_BATCH_RENDERING='/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999'
DATA_DIR_BATCH_SPLATTER_GT_ROOT='/home/xuyimeng/Repo/LGM/runs/splatter_gt'

DATA_DIR_BATCH_RENDERING_SRN='/home/xuyimeng/Data/SRN/srn_cars/cars_train'
DATA_DIR_BATCH_SPLATTER_GT_ROOT_SRN='/home/xuyimeng/Repo/LGM/runs/LGM_optimize_splatter/shapenet_fit_batch'

# [TRAIN VAE]
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# accelerate launch --main_process_port 29510 --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_vae.py big --workspace runs/vae_train/workspace_debug \
#     --lr 1e-6 --num_epochs 20001 --eval_iter 100 --save_iter 100 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --prob_grid_distortion 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'add_code' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --model_type Zero123PlusGaussianVaeKL \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --lambda_kl 1e-6


accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_vae.py big --workspace runs/vae_train/workspace_debug \
    --lr 1e-4 --num_epochs 1 --eval_iter 1 --save_iter 100 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
    --prob_cam_jitter 0 --prob_grid_distortion 0 --input_size 320 --num_input_views 6 --num_views 20 \
    --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
    --desc 'gt_splatter_srn' --data_path_rendering ${DATA_DIR_BATCH_RENDERING_SRN} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT_SRN} \
    --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
    --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
    --model_type Zero123PlusGaussianVaeKL \
    --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
    --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
    --lambda_kl 1e-6 --data_mode srn_cars --overfit_one_scene


# accelerate launch --main_process_port 29510 --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_vae.py big --workspace runs/vae_train/workspace_debug \
#     --lr 1e-4 --num_epochs 20001 --eval_iter 100 --save_iter 100 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --prob_grid_distortion 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'overfit_objaverse' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --model_type Zero123PlusGaussianVaeKL \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder \
#     --lambda_kl 1e-6 --overfit_one_scene

# 4gpus
# accelerate launch --main_process_port 29510 --config_file acc_configs/gpu4.yaml main_zero123plus_v4_batch_code.py big --workspace runs/zerp123plus_batch/workspace_ablation \
#     --lr 1e-4 --num_epochs 20001 --eval_iter 20 --save_iter 20 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --prob_grid_distortion 0 --input_size 320 --num_input_views 6 --num_views 26 \
#     --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 1 \
#     --desc 'ablation_2dot1_fixed_encode_range_4gpus-mix_diffusion_interval_10-resume20240315' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" --codes_from_encoder --mix_diffusion_interval 1 \
#     --resume 'runs/zerp123plus_batch/workspace_ablation/20240315-ablation_2_fixed_encode_range_2gpus_lpips-resume20240312-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0_lpips1.0-lr0.0002-Plat100/eval_epoch_60/model.safetensors'
