#!/bin/bash
#SBATCH --account=b1026
#SBATCH --partition=cosmoscompute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=52
#SBACTH --time=48:00:00
#SBATCH --mem-per-cpu=0
#SBATCH --job-name=chi2_numba
#SBATCH --output=outlogs/chi2_numba_outlog
#SBATCH --error=errlogs/chi2_numba_errlog

module purge all
module load python/anaconda3.6

source activate mtillman_python


for z in 0.0 0.5 1.5 2.0 2.5 2.0 2.5 3.0 3.5 4.0
do 
python multiz_numba_parallel_chi2_SHEN_6param.py $z
done
