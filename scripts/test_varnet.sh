GPU_NUM=0
TEST_CONFIG_YAML="configs/base_varnet,k=1.yaml"
TENSORBOARD_DIR="./results/varnet/folder_2"

CUDA_VISIBLE_DEVICES=$GPU_NUM python test.py \
    --config=$TEST_CONFIG_YAML \
    --batch_size=32 \
    --write_image=1 \
    --tensorboard_dir=$TENSORBOARD_DIR