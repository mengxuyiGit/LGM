# FOLDER_NAME=/mnt/kostas_home/lilym/LGM/LGM/runs/inference/LGM/GSO/workspace_debug/20240926-234659-decoder_inference_gso-cam1.3-loss_render5.0_splatter1.0_lpips5.0-lr0.0001-Plat5/eval_global_step_0
# # python calculate_metrics.py --folder_name $FOLDER_NAME --num_views 21
# python calculate_metrics2.py --folder_name $FOLDER_NAME --num_views 21 --gt_pattern "*_gt.jpg" --pred_pattern "*_pred.jpg" 


FOLDER_NAME=/home/xuyimeng/Repo/Wonder3D/outputs/inference/GSO_metric/nder3D-joint-128-lara_splatter-rope-ZERO_SNR-BSZ16_acc1_gpu4-all_trainable/inference
python calculate_metrics2.py --folder_name $FOLDER_NAME --num_views 21 --gt_pattern "*-gt.jpg" --pred_pattern "*-image-sample_cfg5.0.jpg" --num_views 10