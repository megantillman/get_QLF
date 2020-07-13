#!/bin/bash
#SBATCH --account=b1026
#SBATCH --partition=cosmoshimeme 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=52
#SBACTH --time=48:00:00
#SBATCH --mem-per-cpu=0
#SBATCH --job-name=chi2
#SBATCH --output=outlogs/chi2_outlog
#SBATCH --error=errlogs/chi2_errlog

module purge all
module load python/anaconda3.6

source activate mtillman_python

reso=20
z=1

python parallel_chi2_SHEN_6param.py $reso $z
