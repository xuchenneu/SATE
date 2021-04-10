#! /bin/bash

gpu_num=1

data_dir=
test_subset=test,test1

exp_name=
if [ "$#" -eq 1 ]; then
    exp_name=$1
fi

n_average=10
beam_size=5
max_tokens=20000

cmd="./run.sh
    --stage 2
    --stop_stage 2
    --gpu_num ${gpu_num}
    --exp_name ${exp_name}
    --n_average ${n_average}
    --beam_size ${beam_size}
    --max_tokens ${max_tokens}
    "

if [[ -n ${data_dir} ]]; then
    cmd="$cmd --data_dir ${data_dir}"
fi
if [[ -n ${test_subset} ]]; then
    cmd="$cmd --test_subset ${test_subset}"
fi

echo $cmd
eval $cmd
