# DATA_RENDERING_ROOT_LVIS_46K=/mnt/lingjie_cache/lvis_dataset/testing
# DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT=/mnt/lingjie_cache/lvis_splatters/testing
# DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_CLUSTER=/mnt/kostas-graid/datasets/xuyimeng/lvis/data_processing/testing
DATA_RENDERING_ROOT_LVIS_46K=/home/chenwang/data/lvis_dataset/testing
# DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_FINETUNED_CLUSTER=/mnt/kostas-graid/datasets/xuyimeng/lvis/data_processing_finetuned_lgm_fov60_8epochs/testing
# DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_FINETUNED_CLUSTER=/home/xuyimeng/Repo/zero-1-to-G/runs/lvis/workspace_debug/debug/8000-8999/20240814-213046-load_2dgs_ckpt_save_vis-loss_render1.0_splatter1.0_lpips1.0-lr0.001-Plat/splatters_mv_inference
# DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_FINETUNED_CLUSTER=/home/xuyimeng/Repo/zero-1-to-G/runs/lvis/workspace_debug/debug/8000-8999/20240814-152456-load_2dgs_ckpt_save_vis-loss_render1.0_splatter1.0_lpips1.0-lr0.001-Plat/splatters_mv_inference
# DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_FINETUNED_CLUSTER=/mnt/kostas-graid/datasets/xuyimeng/lara/splatter_data/0/20240824-025018-lara_splatter_6views0_24-epoch2_6000-loss_render1.0_splatter1.0_lpips1.0-lr0.001-Plat/splatters_mv_inference

LVIS_2DGS_SPLATTERS_MV_ROOT=/mnt/kostas-graid/datasets/xuyimeng/lvis/splatter_data_2dgs/testing

# [] DDP of discriminator
# export CUDA_VISIBLE_DEVICES=6,7
accelerate launch --main_process_port 29516 --config_file acc_configs/gpu2.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_accumulate_ddp.py big \
    --workspace runs/finetune_decoder/workspace_decoder_Discriminator2 \
    --lr 5e-6 --num_epochs 10001 --max_train_steps 100000 --eval_iter 100 --save_iter 200 --lr_scheduler Plat \
    --lr_scheduler_patience 5 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
    --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 20 \
    --lambda_splatter 1 --lambda_rendering 5 --lambda_alpha 5 --lambda_lpips 5 \
    --desc 'discriminator_ddp_acc32_GAN_thres0.25_cnt5' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
    --data_path_vae_splatter ${LVIS_2DGS_SPLATTERS_MV_ROOT} \
    --set_random_seed \
    --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
    --custom_pipeline "./zero123plus/pipeline_v8_cat.py" --render_input_views --attr_group_mode "v5" \
    --bg 1.0 --fovy 60 \
    --finetune_decoder \
    --batch_size 2 --num_workers 1 --gradient_accumulation_steps 8 \
    --independent_encode_mv --use_wonder3d_vae --pretrained_model_name_or_path 'lambdalabs/sd-image-variations-diffusers' \
    --disc_conditional --disc_factor 1.0 --discriminator_warm_up_steps 2 --lambda_discriminator 1.0 \
    --invalid_list /mnt/kostas_home/lilym/LGM/LGM/data_lists/lvis_invalid_uids_nineviews.json \
    --resume_decoder /home/chenwang/Repo/LGM/runs/finetune_decoder/workspace_train_sep/00002-LVIS-independent_encode_mv-resume_00012-numV12-loss_render1.5_splatter1.0_lpips2.0-lr5e-06-Plat5/eval_global_step_2000_ckpt/model.safetensors \
    --resume_discriminator /home/chenwang/Repo/LGM/runs/finetune_decoder/workspace_decoder_Discriminator/20240947-discriminator_ddp_acc32_GAN_thres0.25_cnt5-loss_render5.0_splatter1.0_lpips5.0-lr5e-06-Plat5/discriminator/model.safetensors


# # [Sep 24] SD-decoder with finetuned LGM results
# # export CUDA_VISIBLE_DEVICES=3
# accelerate launch --main_process_port 29514 --config_file acc_configs/gpu4.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_accumulate.py big \
#     --workspace runs/finetune_decoder/workspace_train_sep \
#     --lr 5e-6 --num_epochs 10001 --eval_iter 500 --save_iter 1000 --lr_scheduler Plat \
#     --lr_scheduler_patience 5 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --num_input_views 6 --num_views 12 \
#     --lambda_splatter 1 --lambda_rendering 1.5 --lambda_alpha 1 --lambda_lpips 2 --lambda_normal 0.02 --lambda_dist 0.01 \
#     --desc 'LVIS-independent_encode_mv-resume_00012' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
#     --data_path_vae_splatter ${LVIS_2DGS_SPLATTERS_MV_ROOT} \
#     --set_random_seed \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v8_cat.py" --render_input_views --attr_group_mode "v5" \
#     --invalid_list /mnt/kostas_home/lilym/LGM/LGM/data_lists/lvis_invalid_uids_nineviews.json \
#     --bg 1.0 --fovy 60 \
#     --finetune_decoder --use_wonder3d_vae --pretrained_model_name_or_path 'lambdalabs/sd-image-variations-diffusers' \
#     --batch_size 2 --num_workers 1 --gradient_accumulation_steps 2 \
#     --independent_encode_mv \
#     --resume_decoder /home/chenwang/Repo/LGM/runs/finetune_decoder/workspace_train_sep/00001-LVIS-independent_encode_mv-resume_00012-numV12-loss_render1.5_splatter1.0_lpips2.0-lr5e-06-Plat5/eval_global_step_4000_ckpt/model.safetensors

    # --resume_decoder /mnt/kostas_home/lilym/LGM/LGM/runs/finetune_decoder/workspace_train_sep/00012-independent_encode_mv_True-no_resume-GTnormal_loss_after6000-BSZ16-numV15-loss_render1.0_splatter1.0_lpips2.0-lr5e-06-Plat5/eval_global_step_29000_ckpt/model.safetensors
#     # --resume_decoder '/mnt/kostas_home/lilym/LGM/LGM/runs/finetune_decoder/workspace_train_sep/00010-correct_vae_loading-Wonder3D_VAE-256-lara_data-2dgs-GT_normal_loss_after_6000-bsz16-numV15-loss_render1.0_splatter1.0_lpips2.0-lr5e-06-Plat5/eval_global_step_15500_ckpt/model.safetensors'
# #  --output_size 256

# # [Sep 3] SD-decoder with finetuned LGM results
# # export CUDA_VISIBLE_DEVICES=3
# accelerate launch --main_process_port 29514 --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_accumulate.py big \
#     --workspace runs/finetune_decoder/workspace_debug \
#     --lr 5e-6 --num_epochs 10001 --eval_iter 500 --save_iter 1000 --lr_scheduler Plat \
#     --lr_scheduler_patience 5 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 256 --num_input_views 6 --num_views 15 \
#     --lambda_splatter 1 --lambda_rendering 1 --lambda_alpha 1 --lambda_lpips 2 --lambda_normal 0.05 \
#     --desc 'independent_encode_mv_False-no_resume-GTnormal_loss_after6000-BSZ16' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_FINETUNED_CLUSTER} \
#     --set_random_seed \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v8_cat.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 39.6 --data_mode 'lara' \
#     --finetune_decoder --use_wonder3d_vae --pretrained_model_name_or_path 'lambdalabs/sd-image-variations-diffusers' \
#     --batch_size 3 --num_workers 1 --gradient_accumulation_steps 2 \
#     --independent_encode_mv \
#     --resume_decoder '/mnt/kostas_home/lilym/LGM/LGM/runs/finetune_decoder/workspace_train_sep/00010-correct_vae_loading-Wonder3D_VAE-256-lara_data-2dgs-GT_normal_loss_after_6000-bsz16-numV15-loss_render1.0_splatter1.0_lpips2.0-lr5e-06-Plat5/eval_global_step_15500_ckpt/model.safetensors'


# --overfit_one_scene 
    # --resume_decoder '/mnt/kostas_home/lilym/LGM/LGM/runs/finetune_decoder/workspace_train_sep/00008-LARA_DATA-sd_decoder-2dgs-GT_normal_loss_foreground-resume8800-numV15-loss_render1.0_splatter1.0_lpips2.0-lr5e-06-Plat5/eval_global_step_8000_ckpt/model.safetensors' \
    # --resume_decoder /mnt/kostas_home/lilym/LGM/LGM/runs/finetune_decoder/workspace_train_sep/00003-LARA_DATA-sd_decoder-2dgs-only_rendering_loss-bsz12-numV15-loss_render1.0_splatter2.0_lpips2.0-lr5e-06-Plat5/eval_global_step_1600_ckpt/model.safetensors \
    # \
    # --resume_decoder /mnt/kostas_home/lilym/LGM/LGM/runs/finetune_decoder/workspace_train_july/20240714-sd-decoder-fted_lgm-loss_render1.0_splatter2.0_lpips2.0-lr5e-06-Plat5/eval_global_step_2000_ckpt/model.safetensors


# # [Aug 13] 2DGS splatter encode decode
# export CUDA_VISIBLE_DEVICES=3
# accelerate launch --main_process_port 29514 --config_file acc_configs/gpu2.yaml main_zero123plus_v5_batch_marigold_finetune_decoder_accumulate.py big \
#     --workspace runs/finetune_decoder/workspace_train_july \
#     --lr 5e-6 --num_epochs 10001 --eval_iter 500 --save_iter 500 --lr_scheduler Plat \
#     --lr_scheduler_patience 5 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 20 \
#     --lambda_splatter 2 --lambda_rendering 1 --lambda_alpha 1 --lambda_lpips 2 \
#     --desc 'sd-decoder-fted_lgm-resume0713_2k' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_FINETUNED_CLUSTER} \
#     --set_random_seed \
#     --model_type Zero123PlusGaussianMarigoldUnetCrossDomain \
#     --custom_pipeline "./zero123plus/pipeline_v8_cat.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 60 \
#     --finetune_decoder \
#     --batch_size 2 --num_workers 1 --gradient_accumulation_steps 8 \
#     --invalid_list /mnt/kostas_home/lilym/LGM/LGM/data_lists/lvis_invalid_uids_nineviews.json \
#     --resume_decoder /mnt/kostas_home/lilym/LGM/LGM/runs/finetune_decoder/workspace_train_july/20240714-sd-decoder-fted_lgm-loss_render1.0_splatter2.0_lpips2.0-lr5e-06-Plat5/eval_global_step_2000_ckpt/model.safetensors
