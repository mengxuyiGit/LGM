# make sure you are in "conda activate tensorboard"

# LOG_DIR="workspace_debug_pretrained_tf/00049-cam_aligned-inV4-lossV56-lr0.0001"
# LOG_DIR="workspace_debug_pretrained_tf/00074-fix_input_views-inV6-lossV56-lr0.0001" # wrong optimizer
# LOG_DIR="workspace_debug_pretrained_tf/00097-fit_003_no_jitter-inV6-lossV20-lr0.001" # correct optimizer: 6134
# LOG_DIR="workspace_debug_pretrained_tf/00098-fix_inputV_no_jitter-inV6-lossV20-lr0.0001" # 6135
# LOG_DIR="workspace_debug_pretrained_tf/00100-fix_inputV_no_jitter-inV6-lossV20-lr0.001" # 6136
# LOG_DIR="workspace_debug_pretrained_tf/00115-fix_inputV_as1to6-inV6-lossV20-lr0.0006" # 6134
# LOG_DIR="runs/zerp123plus_overfit/workspace_overfit" # 6134
# LOG_DIR="runs/LGM_optimize_splatter/workspace_splatter_gt_full_ply" # 6135
# LOG_DIR="runs/LGM_optimize_splatter/workspace_splatter_gt_full_ply_fixed_einops" # 6135: different lr and lr_scheduler
# LOG_DIR="runs/LGM_optimize_splatter/workspace_debug_batch/00056-batch-Plat-patience_2-factor_0.5-eval_2-adamW-splat128-inV6-lossV20-lr0.003" # 6136
# LOG_DIR="runs/LGM_optimize_splatter/workspace_debug_batch/00059-batch-Plat-patience_2-factor_0.5-eval_5-adamW-splat128-inV6-lossV20-lr0.003/9000-9999" # 6136 ### whole batch, but deferior optimizer
# LOG_DIR="runs/LGM_optimize_splatter/workspace_debug_batch/hydrant-compare/" # 6135
# LOG_DIR="runs/LGM_optimize_splatter/workspace_debug_batch/00063-batch-Plat-patience_2-factor_0.5-eval_50-adamW-splat128-inV6-lossV20-lr0.003/9000-9999" # 6137 ### whole batch, good optimizer
# LOG_DIR="runs/LGM_optimize_splatter/workspace_debug_batch/whole-batch-compare" 
# LOG_DIR="runs/LGM_optimize_splatter/workspace_debug_batch_subset/00004-batch-Plat-patience_2-factor_0.5-eval_50-adamW-subset_-250_-1_splat128-inV6-lossV20-lr0.003" 6138
# LOG_DIR="runs/LGM_optimize_splatter/workspace_debug_batch_subset/compare" # 6139
# cp -r runs/LGM_optimize_splatter/workspace_debug_batch_subset/00009-batch-CosAnn-adamW-subset_2_10_splat128-inV6-lossV20-lr0.0006 runs/LGM_optimize_splatter/workspace_debug_batch_subset/compare/
# LOG_DIR="runs/LGM_optimize_splatter/workspace_debug_batch_subset_es/00000-batch-es10-Plat-patience_2-factor_0.5-eval_5-adamW-subset_10_250_splat128-inV6-lossV20-lr0.003/9000-9999" # 6139

# cp runs/zerp123plus_batch/workspace_train/20240224-194613-train-1gpu-10-250-clamp-norm_scale-skip_predict_x0-loss_render1.0_splatter1.0-lr0.0002/events.out.tfevents.1708803973.kd-a40-0.251073.0 runs/zerp123plus_batch/workspace_train/20240223-164633-train-1gpu-10-250-clamp-norm-norm_scale-skip_predict_x0-loss_render1.0_splatter1.0-lr0.002/continue_20240224-194613-train-1gpu-10-250-clamp-norm_scale-skip_predict_x0-loss_render1.0_splatter1.0-lr0.0002/
# cp runs/zerp123plus_batch/workspace_train/20240224-194635-train-1gpu-10-250-clamp-norm_scale-skip_predict_x0-loss_render1.0-lr0.0002/events.out.tfevents.1708803995.dj-a40-1.grasp.maas.236730.0 runs/zerp123plus_batch/workspace_train/20240223-164633-train-1gpu-10-250-clamp-norm-norm_scale-skip_predict_x0-loss_render1.0_splatter1.0-lr0.002/continue_20240224-194635-train-1gpu-10-250-clamp-norm_scale-skip_predict_x0-loss_render1.0-lr0.0002/
# cp runs/zerp123plus_batch/workspace_train/20240224-203051-train-1gpu-10-250-clamp-skip_predict_x0-loss_render1.0_splatter1.0-lr0.0002/events.out.tfevents.1708806651.dj-a40-1.grasp.maas.239088.0 runs/zerp123plus_batch/workspace_train/20240223-164633-train-1gpu-10-250-clamp-norm-norm_scale-skip_predict_x0-loss_render1.0_splatter1.0-lr0.002/continue_20240224-203051-train-1gpu-10-250-clamp-skip_predict_x0-loss_render1.0_splatter1.0-lr0.0002/
# LOG_DIR="runs/zerp123plus_batch/workspace_train/20240223-164633-train-1gpu-10-250-clamp-norm-norm_scale-skip_predict_x0-loss_render1.0_splatter1.0-lr0.002" # 6137

# LOG_DIR="runs/zerp123plus_batch/workspace_debug/20240225-020445-debug-lr-scheduler-norm_scale-skip_predict_x0-loss_render1.0_splatter1.0-lr0.002"
# LOG_DIR="runs/zerp123plus_batch/workspace_train/20240225-022353-train-920-clamp-skip_predict_x0-loss_render1.0_splatter1.0-lr0.0002-Plat5"
# LOG_DIR="runs/zerp123plus_batch/workspace_train/20240225-051029-train-920-clampv1_fix_rgb-skip_predict_x0-loss_render1.0-lr0.0002-Plat5"
# LOG_DIR="runs/zerp123plus_batch/workspace_train/20240226-002233-train-920-aligned-sf-2gpu-v0_unfreeze_all-skip_predict_x0-loss_render1.0_splatter1.0-lr0.0002-Plat5"
# LOG_DIR="runs/zerp123plus_batch/workspace_train/20240226-012340-train-resume185634-920-aligned-sf-2gpu-v0_unfreeze_all-skip_predict_x0-loss_render1.0_splatter1.0-lr0.0002-Plat5"
# LOG_DIR="runs/zerp123plus_batch/workspace_train/20240226-153220-train-resume012340_epoch310-2gpu-use_ddp_kwargs-v0_unfreeze_all-skip_predict_x0-loss_render1.0-lr0.0002-Plat5"
# LOG_DIR="runs/zerp123plus_batch/workspace_debug/20240229-212338-debug-code-v0_unfreeze_all-skip_predict_x0-loss_render1.0_splatter1.0-lr0.0002-Plat5"
# LOG_DIR="runs/zerp123plus_batch/workspace_train/20240301-093734-optimize-code-splatter-resume021856-sp_guide_2-codes_from_encoder-v0_unfreeze_all-skip_predict_x0-loss_render1.0_splatter1.0-lr0.0002-Plat5"
# LOG_DIR="runs/zerp123plus_batch/workspace_train/20240301-001854-optimize-code-splatter-v0_unfreeze_all-skip_predict_x0-loss_render1.0_splatter1.0-lr0.0002-Plat5"
# LOG_DIR="runs/zerp123plus_batch/workspace_train/20240301-144823-debug-code-from-encoder-sp_guide_1-codes_from_encoder-v0_unfreeze_all-skip_predict_x0-loss_render1.0_splatter1.0-lr0.0002-Plat5"
# LOG_DIR="runs/zerp123plus_batch/workspace_debug_codedir/00001-optimize-code-not-splatter-w-guidance-large-code-lr-sp_guide_10-v0_unfreeze_all-skip_predict_x0-loss_render1.0_splatter1.0-lr0.0002-Plat5"
# LOG_DIR="runs/zerp123plus_batch/workspace_train/20240303-015933-decode_splatter_to_128_upconv-sp_guide_10-codes_lr0.01-v0_unfreeze_all-interpolate_downsample-skip_predict_x0-loss_render1.0_splatter1.0-lr0.0002-Plat5"
# LOG_DIR="runs/zerp123plus_batch/workspace_train/20240302-230033-decode_splatter_to_320-sp_guide_10-codes_lr0.01-v0_unfreeze_all-skip_predict_x0-loss_render1.0_splatter1.0-lr0.0002-Plat5"
# LOG_DIR="runs/zerp123plus_batch/workspace_train/20240303-034021-decode_splatter_to_128_last_layer-sp_guide_10-codes_lr0.01-v0_unfreeze_all-last_layer-skip_predict_x0-loss_render1.0_splatter1.0-lr0.0002-Plat5"
# LOG_DIR="runs/zerp123plus_batch/workspace_train/20240308-4gpus_decode_splatter_to_128_nearest-sp_guide_10-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0-lr0.0002-Plat5"
# LOG_DIR="runs/zerp123plus_batch/workspace_train/20240309-4gpus_decode_splatter_to_128_nearest-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0-lr0.0002-Plat100"
# LOG_DIR="runs/zerp123plus_batch/workspace_train/20240315-4gpus_decode_splatter_to_128_nearest-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter0.5_lpips1.0-lr0.0002-Plat100"
# LOG_DIR="runs/zerp123plus_batch/workspace_unet/20240306-4-gpus-only-latent-loss-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0-lr0.0001-Plat100"
# LOG_DIR="runs/zerp123plus_batch/workspace_unet/20240308-4-gpus-only-latent-loss-resume20240306-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0-lr5e-05-Plat5"
# LOG_DIR="runs/zerp123plus_batch/workspace_unet/20240309-4-gpus-only-latent-loss-resume20240308-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0-lr4e-05-Plat5"
# LOG_DIR="runs/zerp123plus_batch/workspace_unet/20240305-175529-1-gpu-only-latent-loss-overfit_scene2_randinit_unet-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0-lr5e-05-Plat100"
# LOG_DIR="runs/zerp123plus_batch/workspace_unet/20240305-181600-1-gpu-only-latent-loss-overfit_scene2_not_rand_unet-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0-lr5e-05-Plat100"
# LOG_DIR="runs/zerp123plus_batch/workspace_unet/20240315-4-gpus-only-latent-loss-pretrained-unet-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0-lr5e-05-Plat100"
# LOG_DIR="runs/zerp123plus_batch/workspace_unet/20240317-4-gpus-resume-unet20240315-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0-lr1e-05-Plat100"
# LOG_DIR="runs/zerp123plus_batch/workspace_ablation/20240316-ablation4_unet_fixed_encode_range-4gpus-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0-lr1e-05-Plat100"
# LOG_DIR="runs/zerp123plus_batch/workspace_debug/20240322-175501-unet-w-rendering-loss-resume_unet_20240320-sp_guide_1-codes_lr0.01-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-numV6-loss_render1.0_splatter1.0_lpips1.0-lr1e-05-Plat5"
# LOG_DIR="runs/LGM_optimize_splatter/shapenet_fit/00002-debug_srn_cars-es10-Plat-patience_20-factor_0.5-eval_50-adamW-subset_0_2_splat128-inV6-lossV20-lr0.001"
# LOG_DIR="runs/LGM_optimize_splatter/shapenet_fit_batch/00002-debug_srn_cars-es10-Plat-patience_10-factor_0.5-eval_50-adamW-subset_0_2_splat128-inV6-lossV20-lr0.01"
# LOG_DIR="runs/LGM_optimize_splatter/shapenet_fit_batch"
# LOG_DIR="runs/LGM_optimize_splatter/shapenet_test_time"
# LOG_DIR="runs/vae_train/workspace_debug/20240411-204430-add_code-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0_lpips1.0_kl1e-06-lr0.0001-Plat100"
# LOG_DIR="runs/vae_train/workspace_debug/20240412-143548-overfit_srn-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0_lpips1.0_kl1e-06-lr0.0001-Plat100"
# LOG_DIR="runs/vae_train/workspace_train"
# LOG_DIR="runs/vae_train/workspace_train/00004-ddp_srn-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0_lpips1.0_kl1e-06-lr0.0001-Plat100"
# LOG_DIR="runs/zerp123plus_batch/workspace_debug/20240418-155022-ablation_2_only_decoder_srn_overfit_mask_loss-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0_lpips1.0-lr0.0001-Plat100"
# LOG_DIR="runs/zerp123plus_batch/workspace_debug/20240418-155502-ablation_2_only_decoder_objaverse-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0_lpips1.0-lr0.0001-Plat100"
LOG_DIR="runs/zerp123plus_batch/workspace_debug"

TB_PORT=6138
IP_ADDRESS=$(hostname -I | cut -d' ' -f1)

TB_FOLDER=$1

echo "Go to http://$IP_ADDRESS:$TB_PORT"

# Load TensorBoard and start it on the chosen port
tensorboard --logdir="${LOG_DIR}" --host=localhost --port=${TB_PORT} &

## on the local computer: # ssh -N -f -L localhost:<local_port, not in this file>:localhost:<TB_PORT> xuyimeng@chiron
# ssh -N -f -L localhost:16023:localhost:6130 xuyimeng@chiron
## and open "localhost:16023"