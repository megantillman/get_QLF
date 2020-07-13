#!/bin/bash
#SBATCH --account=b1026
#SBATCH --partition=cosmoscompute 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu= 150M
#SBATCH --job-name=QLF_emcee_indz3
#SBATCH --output=outlogs/emcee_indz3_outlog
#SBATCH --error=errlogs/emcee_indz3_errlog

module purge all
module load python/anaconda3.6

source activate mtillman_python

pids=()

for i in 0.0 0.1 0.2 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 
do 
    python -W ignore run_emcee_1z_recur_3param.py $i &
    pids=(${pids[@]} $!)
done

# wait for all pids
for pid in ${pids[*]}
do
    wait $pid
done
