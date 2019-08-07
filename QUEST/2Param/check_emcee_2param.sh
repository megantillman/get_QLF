#!/bin/bash
#SBATCH --account=b1026
#SBATCH --partition=cosmoscompute 
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24
#SBATCH --time=36:00:00
#SBATCH --mem-per-cpu= 150M
#SBATCH --job-name=please_emcee
#SBATCH --output=outlogs/please_emcee_outlog
#SBATCH --error=errlogs/please_emcee_errlog

module purge all
module load python/anaconda3.6

source activate mtillman_python

python -W ignore check_emcee_2param.py 1.5
