#!/bin/bash
#SBATCH --account=b1026  ## <-- EDIT THIS TO BE YOUR ALLOCATION
#SBATCH --partition=cosmoscompute  ## <-- EDIT THIS TO BE YOUR QUEUE NAME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=2G
#SBATCH --job-name=QLF_emcee
#SBATCH --output=outlogs/QLF_emcee_outlog
#SBATCH --error=errlogs/QLF_emcee_errlog

module purge all
module load python/anaconda3.6

source activate mtillman_python

python -W ignore emcee_run_dM.py
