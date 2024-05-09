#!/usr/bin/env bash
#SBATCH -A NAISS2023-22-607
#SBATCH -p alvis
#SBATCH --gpus-per-node=T4:1  -t 16:00:00
#SBATCH -o preprocessing_50.out
#SBATCH -e preprocessing_50.err

# python3 -m venv .recbench_env
source .recbench_env/bin/activate
# pip3 install -r requirements.txt

srun python ./src/main.py
