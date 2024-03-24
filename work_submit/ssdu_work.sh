#!/bin/bash
#SBATCH --job-name=ssdu
#SBATCH --output=ssdu_block17.txt
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --partition=a100 


module load python/3.8-anaconda

conda activate modl

bash scripts/train_ssdu_k1.sh
bash scripts/train_ssdu_k2.sh
bash scripts/train_ssdu_k3.sh
bash scripts/train_ssdu_k4.sh
bash scripts/train_ssdu_k5.sh
bash scripts/train_ssdu_k6.sh
bash scripts/train_ssdu_k7.sh
bash scripts/train_ssdu_k8.sh
bash scripts/train_ssdu_k9.sh
bash scripts/train_ssdu_k10.sh
# bash scripts/train_ssdu_trival.sh


bash scripts/test_ssdu_k1.sh
bash scripts/test_ssdu_k2.sh
bash scripts/test_ssdu_k3.sh
bash scripts/test_ssdu_k4.sh
bash scripts/test_ssdu_k5.sh
bash scripts/test_ssdu_k6.sh
bash scripts/test_ssdu_k7.sh
bash scripts/test_ssdu_k8.sh
bash scripts/test_ssdu_k9.sh
bash scripts/test_ssdu_k10.sh
# python train.py --config configs/base_modl,k=1.yaml
# python test.py --config configs/base_modl,k=1.yaml