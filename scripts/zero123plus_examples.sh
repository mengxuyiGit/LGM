DATA_RENDERING_ROOT_LVIS_46K=/mnt/lingjie_cache/lvis_dataset/testing
DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT=/mnt/kostas-graid/datasets/xuyimeng/lvis/data_processing_finetuned_lgm_fov60_8epochs/testing
# DATA_RENDERING_ROOT_LVIS_46K_CLUSTER=/home/chenwang/data/lvis_dataset/testing
# DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT_CLUSTER=/mnt/kostas-graid/datasets/xuyimeng/lvis/data_processing_finetuned_lgm_fov60_8epochs/testing

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
# export CUDA_VISIBLE_DEVICES=6
accelerate launch --main_process_port 29518 --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_finetune_0123++_low_res.py big \
    --workspace runs/finetune_0123++/workspace_debug \
    --lr 1e-5 --max_train_steps 100000 --eval_iter 500 --save_iter 1000 --lr_scheduler Plat \
    --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
    --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 6 \
    --lambda_splatter 0 --lambda_rendering 0 --lambda_alpha 0 --lambda_lpips 0 --lambda_splatter_lpips 0 \
    --desc 'res320-b4_g6_acc2_encode_decode' --data_path_rendering ${DATA_RENDERING_ROOT_LVIS_46K} \
    --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
    --set_random_seed \
    --model_type Zero123PlusLowRes \
    --custom_pipeline "./zero123plus/pipeline_v8_cat.py" --render_input_views --attr_group_mode "v5" \
    --bg 1.0 --fovy 60 --rendering_loss_use_weight_t \
    --train_unet --class_emb_cat  --drop_cond_prob 0.1 --only_train_attention \
    --batch_size 2 --num_workers 1 --gradient_accumulation_steps 2 \
    --invalid_list /mnt/kostas_home/lilym/LGM/LGM/data_lists/lvis_invalid_uids_nineviews.json \
    --train_unet_single_attr input 
    # --overfit_one_scene

    # --resume_unet /mnt/kostas_home/lilym/LGM/LGM/runs/finetune_unet/workspace_train_july/00007-train_unet-sd_encoder-fted_LGM_fov60-NO_resume-train_unet-numV10-loss-lr4e-06-Plat50/eval_global_step_2000_ckpt/model.safetensors \
    # --resume_decoder /mnt/kostas_home/lilym/LGM/LGM/runs/finetune_decoder/workspace_train_july/20240713-sd-decoder-fted_lgm-loss_render1.0_splatter2.0_lpips2.0-lr0.0001-Plat5/eval_global_step_1000_ckpt/model.safetensors

# # [INFERENCE] 
# # DATA_RENDERING_ROOT_GSO=/home/xuyimeng/Data/gso/liuyuan/view1 
# DATA_RENDERING_ROOT_GSO=/mnt/kostas-graid/sw/envs/chenwang/data/gso/gso_recon_gsec512
# accelerate launch --main_process_port 29518 --config_file acc_configs/gpu1.yaml main_zero123plus_v5_batch_marigold_finetune_0123++_inference_GSO.py big \
#     --workspace runs/finetune_0123++/workspace_inference \
#     --lr 4e-6 --max_train_steps 100000 --eval_iter 500 --save_iter 1000 --lr_scheduler Plat \
#     --lr_scheduler_patience 50 --lr_scheduler_factor 0.7 --lr_schedule_by_train \
#     --prob_cam_jitter 0 --input_size 320 --output_size 320 --num_input_views 6 --num_views 6 \
#     --lambda_splatter 0 --lambda_rendering 0 --lambda_alpha 0 --lambda_lpips 0 --lambda_splatter_lpips 0 \
#     --desc 'inference-8k-encode_decode_rgb320' --data_path_rendering ${DATA_RENDERING_ROOT_GSO} --metric_GSO \
#     --data_path_vae_splatter ${DATA_DIR_BATCH_LVIS_SPLATTERS_MV_ROOT} \
#     --set_random_seed \
#     --model_type Zero123PlusLowRes \
#     --custom_pipeline "./zero123plus/pipeline_v8_cat.py" --render_input_views --attr_group_mode "v5" \
#     --bg 1.0 --fovy 60 --rendering_loss_use_weight_t \
#     --inference_finetuned_unet --class_emb_cat  --drop_cond_prob 0.1 --only_train_attention \
#     --batch_size 1 --num_workers 1 --gradient_accumulation_steps 1 \
#     --invalid_list /mnt/kostas_home/lilym/LGM/LGM/data_lists/lvis_invalid_uids_nineviews.json \
#     --train_unet_single_attr input \
#     --resume_unet runs/finetune_0123++/workspace_train_july/00001-res320-b4_g6_acc2_encode_decode-train_unet-numV6-loss-lr1e-05-Plat50/eval_global_step_8000_ckpt/model.safetensors

#     # --resume_unet /mnt/kostas_home/lilym/LGM/LGM/runs/finetune_0123++/workspace_train_july/00000-b12_g3_encode_decode_rgb128-train_unet-numV6-loss-lr4e-06-Plat50/eval_global_step_41000_ckpt/model.safetensors
    