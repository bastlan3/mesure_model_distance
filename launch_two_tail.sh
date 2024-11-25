#!/bin/bash

#SBATCH -t 5:00:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:3,vram:15G 
#SBATCH -c 8
#SBATCH --mem 50G
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=bastien_lelan@hms.harvard.edu

module load gcc/9.2.0
module load cuda/11.7
module load python/3.9.14

source /home/bal323/monkey_lab/bin/activate

export WANDB_API_KEY= de19c5a1d964820317c7733e7aca31fda17e4acd

/n/cluster/bin/job_gpu_monitor.sh &
python3 /home/bal323/model_learn_distance/two_tailed_model.py
