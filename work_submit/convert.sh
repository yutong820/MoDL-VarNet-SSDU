#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --job-name=convert
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=20:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
module load python/3.8-anaconda
# conda init bash
eval "$(conda shell.bash hook)"
conda activate modl

python convert.py