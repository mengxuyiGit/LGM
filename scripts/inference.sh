DATA_DIR_BATCH_RENDERING='/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets/9000-9999'
DATA_DIR_BATCH_SPLATTER_GT_ROOT='/home/xuyimeng/Repo/LGM/runs/splatter_gt'

# python infer.py big --resume pretrained/model_fp16.safetensors --workspace workspace_test/0123 \
#     --test_path ${DATA_DIR_BATCH_RENDERING}/ffb0d644238b4c679658aa0ee46ac6da


python infer_zero123plus.py big --resume pretrained/model_fp16.safetensors --workspace workspace_test/0123/srn_car_yellow_front \
    --test_path ${DATA_DIR_BATCH_RENDERING}/ffb0d644238b4c679658aa0ee46ac6da \
    --num_input_views 6 --model_type LGM


# python infer_cw.py big --resume pretrained/model_fp16.safetensors --workspace workspace_test/cw \
#     --test_path ${DATA_DIR_BATCH_RENDERING}/ffb0d644238b4c679658aa0ee46ac6da \
#     --num_input_views 6 --model_type LGM


# # [INFERENCE with diffusion]
# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference.py big --workspace runs/zerp123plus_batch/workspace_inference \
#     --lr 2e-4 --num_epochs 10001 --eval_iter 10 --save_iter 10 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 0 \
#     --desc 'ablation4_unet_20240320_settimesteps' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interva   l 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_diffusion --one_step_diffusion 1000 \
#     --resume "runs/zerp123plus_batch/workspace_ablation/20240320-ablation4_unet_fixed_encode_range-3gpus-resume_ep120-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0-lr4e-06-Plat100/eval_epoch_140/model.safetensors"
#     # --resume "runs/zerp123plus_batch/workspace_ablation/20240316-ablation4_unet_fixed_encode_range-4gpus-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0-lr1e-05-Plat100/eval_epoch_120/model.safetensors"


# # [INFERENCE ORIGIANL ZERO123plus with diffusion]
# export CUDA_VISIBLE_DEVICES=2
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference.py big --workspace runs/zerp123plus_batch/workspace_debug \
#     --lr 2e-4 --num_epochs 10001 --eval_iter 10 --save_iter 10 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 0 \
#     --desc 'decoder_no_in_ckpt' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_cache \
#     --code_cache_dir "runs/zerp123plus_batch/workspace_ablation/20240314-ablation4-fixed-encode-range-4gpus-resume-e30+180-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0_lpips1.0-lr0.0002-Plat100/code_dir" \
#     --resume "runs/zerp123plus_batch/workspace_ablation/20240320-ablation4_unet_fixed_encode_range-3gpus-resume_ep120-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0-lr4e-06-Plat100/eval_epoch_2000/model.safetensors"
#     # --resume "runs/zerp123plus_batch/workspace_ablation/20240321-ablation4_unet_fixed_encode_range-4gpus-resumeunet20240320_ep140-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0-lr2e-06-Plat100/eval_epoch_620/model.safetensors"
#     # --resume "runs/zerp123plus_batch/workspace_ablation/20240320-ablation4_unet_fixed_encode_range-3gpus-resume_ep120-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0-lr4e-06-Plat100/eval_epoch_140/model.safetensors"
#     # --resume "runs/zerp123plus_batch/workspace_ablation/20240316-ablation4_unet_fixed_encode_range-4gpus-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0-lr1e-05-Plat100/eval_epoch_120/model.safetensors"

# # [INFERENCE LoRA] NOTE
# # export CUDA_VISIBLE_DEVICES=1
# accelerate launch --config_file acc_configs/gpu1.yaml main_zero123plus_v4_batch_code_inference.py big --workspace runs/zerp123plus_batch/workspace_inference2 \
#     --lr 2e-4 --num_epochs 10001 --eval_iter 5 --save_iter 5 --lr_scheduler Plat --lr_scheduler_patience 100 --lr_scheduler_factor 0.7 \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 0 --lambda_rendering 1 --lambda_alpha 0 --lambda_lpips 0 \
#     --desc 'debug_encode_splatter' --data_path_rendering ${DATA_DIR_BATCH_RENDERING} --data_path_splatter_gt ${DATA_DIR_BATCH_SPLATTER_GT_ROOT} \
#     --set_random_seed --batch_size 1 --num_workers 1 --plot_attribute_histgram 'scale' \
#     --skip_predict_x0 --scale_act 'biased_softplus' --scale_act_bias -3 --scale_bias_learnable \
#     --scale_clamp_max -2 --scale_clamp_min -10 --model_type Zero123PlusGaussianCode \
#     --splatter_guidance_interval 1 --save_train_pred -1 --decode_splatter_to_128 \
#     --decoder_upblocks_interpolate_mode "last_layer" \
#     --codes_from_diffusion --vae_on_splatter_image 

    # --resume "runs/zerp123plus_batch/workspace_ablation/20240322-175501-unet-w-rendering-loss-resume_unet_20240320-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-numV6-loss_render1.0_splatter1.0_lpips1.0-lr1e-05-Plat5/model.safetensors"

    # \

    # --resume 'runs/zerp123plus_batch/workspace_ablation/20240315-ablation_2_fixed_encode_range_2gpus_lpips-resume20240312-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0_lpips1.0-lr0.0002-Plat100/eval_epoch_60/model.safetensors'
    # --resume "runs/zerp123plus_batch/workspace_debug/20240327-165142-ablation_2_fixed_encode_range_overfit_scene2-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0_lpips1.0-lr0.0001-Plat100/model.safetensors"
    # # "runs/zerp123plus_batch/workspace_ablation/20240328-ablation_2dot1_fixed_encode_range_4gpus-mix_diffusion_interval_10-resume20240315-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-numV26-loss_render1.0_splatter1.0_lpips1.0-lr0.0001-Plat100/model.safetensors"
    # 'runs/zerp123plus_batch/workspace_ablation/20240312-ablation_2_fixed_encode_range_2gpus_lpips-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0_lpips1.0-lr0.0002-Plat100/eval_epoch_40/model.safetensors'
    # "runs/zerp123plus_batch/workspace_ablation/20240326-134858-ablation_2_fixed_encode_range_overfit_scene2-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0_lpips1.0-lr0.0001-Plat100/model.safetensors"
    # --resume "runs/zerp123plus_batch/workspace_ablation/20240322-175501-unet-w-rendering-loss-resume_unet_20240320-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-numV6-loss_render1.0_splatter1.0_lpips1.0-lr1e-05-Plat5/model.safetensors"
    # --lipschitz_coefficient 0.6 --lipschitz_mode gaussian_noise 
    # \
    # --code_cache_dir "runs/zerp123plus_batch/workspace_ablation/20240314-ablation4-fixed-encode-range-4gpus-resume-e30+180-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0_lpips1.0-lr0.0002-Plat100/code_dir" \
    # --resume "runs/zerp123plus_batch/workspace_debug/20240322-175501-unet-w-rendering-loss-resume_unet_20240320-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-numV6-loss_render1.0_splatter1.0_lpips1.0-lr1e-05-Plat5/eval_epoch_110/model.safetensors"
    # # --resume "runs/zerp123plus_batch/workspace_ablation/20240322-195548-unet-w-rendering-loss-resume_unet_w_rendering20240322-022336-epo30-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-numV6-loss_render1.0_splatter1.0_lpips1.0-lr1e-05-Plat5/eval_epoch_90/model.safetensors"
    # # --resume "runs/zerp123plus_batch/workspace_ablation/20240322-022336-unet-w-rendering-loss-resume_unet_20240320-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-numV6-loss_render1.0_splatter1.0_lpips1.0-lr1e-05-Plat5/eval_epoch_30/model.safetensors"
    # # --resume "runs/zerp123plus_batch/workspace_lora/00002-lora-lora_rank128-gpus4-sp_guide_1-loss_render1.0-lr1e-05-Plat100/eval_epoch_140/model.safetensors"
