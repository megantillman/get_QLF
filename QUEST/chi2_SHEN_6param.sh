#!/bin/bash
#SBATCH --account=b1026
#SBATCH --partition=cosmoscompute 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=52
#SBATCH --job-name=chi2_SHEN_6param
#SBATCH --output=outlogs/chi2_SHEN_6param_outlog
#SBATCH --error=errlogs/chi2_SHEN_6param_errlog

module purge all
module load python/anaconda3.6

source activate mtillman_python

reso=20
div=1230800
iters=$((reso**6/div))
z=1

python combos_SHEN_6param.py $reso $z

for index in $(seq 0 $iters)
do 
    python slices_SHEN_6param.py $index $reso $div $z $iters &
    pids=(${pids[@]} $!)
done

# wait for all pids
for pid in ${pids[*]}
do
    wait $pid
done

python combine_SHEN_6param.py $iters $reso $div
