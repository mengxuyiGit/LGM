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
# LOG_DIR="runs/zerp123plus_batch/workspace_debug"
# LOG_DIR="/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold/workspace_train/tensorboard-compare"
# LOG_DIR="/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold"
# LOG_DIR="/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold/workspace_test/reg-both-encode-save-iters-decode-20240424-184835-v5_LGM_init_scene1andAfter_to_decoded_png_inference1000-vae_on_splatter_image-codes_from_diffusion-loss_render1.0_lpips1.0-lr0.002-Plat/zero123plus"
# LOG_DIR="/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold/workspace_debug/20240425-003322-v5_LGM_init_scene0andAfter_reg_encoder_input_every_iter_no_clip-vae_on_splatter_image-codes_from_diffusion-loss_render1.0_lpips1.0-lr0.002-Plat"
# LOG_DIR="/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold/workspace_debug/20240425-021703-v5_LGM_init_srn_cars_reg_encoder_input_every_iter_no_clip-vae_on_splatter_image-codes_from_diffusion-loss_render1.0_lpips1.0-lr0.002-Plat"
# LOG_DIR="/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold/workspace_debug/20240425-033032-v5_LGM_320_init_scene0andAfter_reg_encoder_input_every_iter_no_clip-vae_on_splatter_image-codes_from_diffusion-loss_render1.0_lpips1.0-lr0.002-Plat"
# LOG_DIR="/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold/workspace_debug/20240425-185946-v5_LGM_init_scene0andAfter_reg_encoder_input_every_iter_no_clip-vae_on_splatter_image-codes_from_diffusion-loss_render1.0_lpips1.0-lr0.002-Plat"
# LOG_DIR="runs/marigold/workspace_test/20240425-205821-v5_LGM_init_render320_scene_0_100_reg_encoder_input_every_iter_no_clip-vae_on_splatter_image-codes_from_diffusion-loss_render1.0_lpips1.0-lr0.002-Plat"
# LOG_DIR="runs/marigold/workspace_debug/20240425-211335-v5_LGM_init_render320_scene_0_200_reg_encoder_input_every_iter_no_clip-vae_on_splatter_image-codes_from_diffusion-loss_render1.0_lpips1.0-lr0.002-Plat"
# LOG_DIR="/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold/workspace_optimize"
# LOG_DIR="/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold/workspace_test/reg-both-encode-every-iter-20240425-003322-v5_LGM_init_scene0andAfter_reg_encoder_input_every_iter_no_clip-vae_on_splatter_image-codes_from_diffusion-loss_render1.0_lpips1.0-lr0.002-Plat"
# LOG_DIR="/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold/workspace_test/reg-both-encode-save-iters-decode-20240424-184835-v5_LGM_init_scene1andAfter_to_decoded_png_inference1000-vae_on_splatter_image-codes_from_diffusion-loss_render1.0_lpips1.0-lr0.002-Plat"
# LOG_DIR="runs/marigold_srn"
# LOG_DIR="runs/marigold_unet/workspace_debug/20240429-013821-marigold-unet-w-rendering-loss-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0_lpips1.0-lr1e-05-Plat5"
# LOG_DIR="runs/marigold_unet/workspace_debug/20240429-014831-marigold-unet-w-rendering-loss-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0_lpips1.0-lr1e-06-Plat5"
# LOG_DIR="runs/marigold_unet/workspace_debug/20240429-020506-marigold-unet-w-rendering-loss-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter1.0_lpips1.0-lr1e-05-Plat5"
# LOG_DIR="runs/marigold_unet/workspace_debug/20240429-023105-marigold-unet-wo-rendering-loss-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss-lr1e-06-Plat5"
# LOG_DIR="/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold_unet/workspace_CD_ovft"
# LOG_DIR="runs/marigold_unet/workspace_CD_debug/20240515-010023-debug_loss_curve-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_lpips1.0-lr0.0001-Plat5"
# LOG_DIR="runs/marigold_unet/workspace_CD_debug/20240515-012622-debug_loss_curve-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_lpips1.0-lr0.0001-Plat5"
# LOG_DIR="/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/marigold_unet/workspace_CD_train/20240519-LARGE-LR-Attn-only_attn-rendering_w_t-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_lpips1.0-lr0.0001-Plat5"
# LOG_DIR="/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_decoder/workspace_train/00000-6gpus_bsz2_accumulate20-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_lpips1.0-lr0.0001-Plat50"
# LOG_DIR="runs/finetune_decoder/workspace_debug/20240522-174900-debug_global_step_tensorboard-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_lpips1.0-lr0.0001-Plat5"
# LOG_DIR="runs/finetune_decoder/workspace_debug/20240522-182637-debug_global_step_tensorboard-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_lpips1.0-lr0.0001-Plat5"
# LOG_DIR="runs/finetune_decoder/workspace_debug/20240522-213058-debug_global_step_tensorboard-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_lpips1.0-lr0.0001-Plat5"
# LOG_DIR="runs/finetune_decoder/workspace_debug/20240523-062407-debug_global_step_tensorboard-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_lpips1.0-lr0.0001-Plat5"
# LOG_DIR=/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_decoder/workspace_train/00003-4gpus_bsz2_accumulate32-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_lpips1.0-lr0.0001-Plat50
# LOG_DIR=/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_decoder/workspace_debug/20240523-180126-debug_tb_log_splatter-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-skip_predict_x0-loss_render1.0_splatter100.0_lpips1.0-lr0.0001-Plat5
# LOG_DIR=/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_unet/workspace_train/00002-train_unet-4gpus_bsz2_accumulate32-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-train_unet-skip_predict_x0-loss_render1.0_lpips1.0-lr3e-05-Plat50
# LOG_DIR=/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_decoder/workspace_train
# LOG_DIR=/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_unet/workspace_train
# LOG_DIR=/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_unet/workspace_train/20240533-train_unet_SPLATTER_LOSS-resume_22000ckpt_ALL_ATTR_pipev7_CD_attn_no_pos_embed_4gpus_bsz2_accumulate2-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-train_unet-loss_render1.0_splatter1.0_lpips1.0-lr3e-05-Plat50
# LOG_DIR=/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_unet/workspace_debug/20240529-235019-train_unet_ALL_LAYERS_pipev7_6gpus_bsz2_accumulate2-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-train_unet-loss_render1.0_lpips1.0-lr3e-05-Plat50
# LOG_DIR=/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/runs/finetune_unet/workspace_train/20240554-rendering_loss-only_train_attention-v7-sp_guide_1-codes_from_encoder-v0_unfreeze_all-pred128_last_layer-train_unet-loss_render1.0_lpips1.0-lr1e-05-Plat50
# LOG_DIR=/mnt/kostas_home/lilym/LGM/LGM/runs/finetune_unet/workspace_train_june
# LOG_DIR=/mnt/kostas_home/lilym/LGM/LGM/runs/finetune_unet/workspace_overfit_june
LOG_DIR=/mnt/kostas_home/lilym/LGM/LGM/runs/finetune_decoder/workspace_train_july

TB_PORT=6146
IP_ADDRESS=$(hostname -I | cut -d' ' -f1)

TB_FOLDER=$1

echo "Go to http://$IP_ADDRESS:$TB_PORT"

# Load TensorBoard and start it on the chosen port
tensorboard --logdir="${LOG_DIR}" --host=localhost --port=${TB_PORT} &

## on the local computer: # ssh -N -f -L localhost:<local_port, not in this file>:localhost:<TB_PORT> xuyimeng@chiron
# ssh -N -f -L localhost:16023:localhost:6130 xuyimeng@chiron
## and open "localhost:16023"