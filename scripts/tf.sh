# LOG_DIR=/mnt/kostas_home/lilym/LGM/LGM/runs/finetune_lgm/workspace_train
# LOG_DIR=/mnt/kostas_home/lilym/LGM/LGM/runs/finetune_refinenet/workspace_train_june
# LOG_DIR=/home/xuyimeng/Repo/2d-gaussian-splatting/output/nerf_synthetic/lego
LOG_DIR=/mnt/kostas_home/lilym/LGM/LGM/runs/finetune_lgm/workspace_ovft

TB_PORT=6138
IP_ADDRESS=$(hostname -I | cut -d' ' -f1)

TB_FOLDER=$1

echo "Go to http://$IP_ADDRESS:$TB_PORT"

# Load TensorBoard and start it on the chosen port
tensorboard --logdir="${LOG_DIR}" --host=localhost --port=${TB_PORT} &

## on the local computer: # ssh -N -f -L localhost:<local_port, not in this file>:localhost:<TB_PORT> xuyimeng@chiron
# ssh -N -f -L localhost:16023:localhost:6130 xuyimeng@chiron
## and open "localhost:16023"