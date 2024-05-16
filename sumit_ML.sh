#!/bin/sh
#SBATCH --output=ML.out
#SBATCH --job-name=ML
#SBATCH --nodes=1
##SBATCH --cpus-per-task=1
#SBATCH --partition=All
##SBATCH --ntasks-per-node=12
##SBATCH --exclude=node23

python LGSPSO_kfold_SVR.py
