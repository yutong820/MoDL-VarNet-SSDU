#!/bin/bash
#SBATCH --job-name=modl
#SBATCH --output=work_output_modl.txt
#SBATCH --gres=gpu:a100:1
#SBATCH --time=20:00:00
#SBATCH --partition=a100  


module load python/3.8-anaconda

conda activate modl

bash scripts/train.sh
bash scripts/test.sh
# python train.py --config configs/base_modl,k=1.yaml
# python test.py --config configs/base_modl,k=1.yaml