#!/bin/bash
## SLURM scripts have a specific format. 

## job name
#SBATCH --job-name=text_inversion
#SBATCH --output=/srv/share4/shalbe3/text_inversion/output/%j-%a.out
#SBATCH --error=/srv/share4/shalbe3/text_inversion/error/%j-%a.err

## partition name
#SBATCH --partition=debug
## number of nodes
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --constraint="a40"
## number of tasks per node
#SBATCH --ntasks-per-node=1

srun --label wrapper.sh