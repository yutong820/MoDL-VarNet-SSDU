GPU_NUM=0
TRAIN_CONFIG_YAML="configs/base_varnet,k=10.yaml"
TENSORBOARD_DIR="./results/varnet/folder_4"

CUDA_VISIBLE_DEVICES=$GPU_NUM python train.py \
    --config=$TRAIN_CONFIG_YAML \
    --write_image=10 \
    --tensorboard_dir=$TENSORBOARD_DIR