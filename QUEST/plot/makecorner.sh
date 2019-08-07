/#!/bin/bash
#SBATCH --account=b1026
#SBATCH --partition=cosmoscompute 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu= 10M
#SBATCH --job-name=corner
#SBATCH --output=outlogs/corner_outlog
#SBATCH --error=errlogs/corner_errlog

module purge all
module load python/anaconda3.6

source activate mtillman_python

python makecorner.py "/projects/b1026/mtillman/QLF_emcee/output/check_emcee_chain-data_2Param_z=1.5.h5py" 2 1.5check
