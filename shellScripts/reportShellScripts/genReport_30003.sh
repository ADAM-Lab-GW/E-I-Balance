#!/bin/bash 

#SBATCH -o LOG%j.out
#SBATCH -e LOG%j.out
#SBATCH -p nano
#SBATCH -N 1
#SBATCH -D /lustre/groups/adamgrp/joey-ICONS-2023/surrogate-learning
#SBATCH -J Surr_training
#SBATCH --export=NONE
#SBATCH -t 25:00
#SBATCH --nice=100

module load python3/3.7.2
python3.7 scripts/generateReport.py -r sparse080_10class_110_t2_e2.html -s sparse080_10class -t 110 111 -a  -d  -e 2
