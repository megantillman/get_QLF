reso=2
div=33
iters=$((reso**6/div))
z=1

python3 combos_SHEN_6param.py $reso $z

for index in $(seq 0 $iters)
do 
    python3 slices_SHEN_6param.py $index $reso $div $z $iters &
    pids=(${pids[@]} $!)
done

# wait for all pids
for pid in ${pids[*]}
do
    wait $pid
done

python3 combine_SHEN_6param.py $iters $reso $div