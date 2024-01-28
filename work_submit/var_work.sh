#!/bin/bash
#SBATCH --job-name=varnet
#SBATCH --output=work_output_varnet.txt
#SBATCH --gres=gpu:a100:1
#SBATCH --time=20:00:00
#SBATCH --partition=a100  


module load python/3.8-anaconda

conda activate modl

bash scripts/train_varnet.sh
bash scripts/test_varnet.sh
# python train.py --config configs/base_varnet,k=1.yaml
# python test.py --config configs/base_varnet,k=1.yaml