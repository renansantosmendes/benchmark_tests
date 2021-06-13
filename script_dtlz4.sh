#!/bin/bash 
#SBATCH --qos=qos-7d
#SBATCH --partition=sorai-a
#SBATCH --output=jobArray_%A_%a.out
#SBATCH --error=jobArray_%A_%a.err
module load python3.7
python DTLZ1.py