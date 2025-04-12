#!/bin/bash 

#SBATCH -o LOG%j.out
#SBATCH -e LOG%j.out
#SBATCH -p nano
#SBATCH -N 1
#SBATCH -D /lustre/groups/adamgrp/joey-ICONS-2023/surrogate-learning
#SBATCH -J Surr_training
#SBATCH --export=NONE
#SBATCH -t 29:59
#SBATCH --nice=100

module load python3/3.7.2
python3.7 scripts/generateReport.py -r sparse050_10class_147_t7_e-1.html -s sparse050_10class -t 147 148 -a  -d  -e -1