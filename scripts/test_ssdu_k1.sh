GPU_NUM=0
TEST_CONFIG_YAML="configs/base_ssdu,k=1.yaml"
TENSORBOARD_DIR="./results/ssdu/block17"

CUDA_VISIBLE_DEVICES=$GPU_NUM python test_ssdu.py \
    --config=$TEST_CONFIG_YAML \
    --batch_size=1 \
    --write_image=1 \
    --tensorboard_dir=$TENSORBOARD_DIR