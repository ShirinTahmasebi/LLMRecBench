#!/usr/bin/env bash
#SBATCH -A NAISS2023-22-607
#SBATCH -p alvis
#SBATCH --gpus-per-node=A100:1  -t 16:00:00
#SBATCH --job-name=preprocessing
#SBATCH --output=preprocessing_%j.out
#SBATCH --error=preprocessing_%j.err

# python3 -m venv .recbench_env
source .recbench_env/bin/activate
# pip3 install -r requirements.txt

srun python ./src/inference.py --model_name "GenRec" --dataset_name "AMAZON_TOY_GAMES" --start_index 0 --end_index 600
