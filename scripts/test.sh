GPU_NUM=0
TEST_CONFIG_YAML="configs/base_modl,k=10.yaml"
TENSORBOARD_DIR="./results/modl/folder_4"

CUDA_VISIBLE_DEVICES=$GPU_NUM python test.py \
    --config=$TEST_CONFIG_YAML \
    --batch_size=32 \
    --write_image=1 \
    --tensorboard_dir=$TENSORBOARD_DIR