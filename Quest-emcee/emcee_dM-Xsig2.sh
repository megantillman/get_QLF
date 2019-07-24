#!/bin/bash
#SBATCH --account=b1026  ## <-- EDIT THIS TO BE YOUR ALLOCATION
#SBATCH --partition=cosmoscompute  ## <-- EDIT THIS TO BE YOUR QUEUE NAME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=300M
#SBATCH --job-name=QLF_emcee
#SBATCH --output=outlogs/emcee2_outlog
#SBATCH --error=errlogs/emcee2_errlog

module purge all
module load python/anaconda3.6

source activate mtillman_python

python -W ignore emcee_run_dM-Xsig2.py
