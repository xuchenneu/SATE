#! /bin/bash

# training the model

gpu_num=1

test_subset=(tst-COMMON)
exp_name=test

n_average=10
beam_size=5
max_tokens=40000

cmd="./run.sh
    --stage 3
    --stop_stage 3
    --gpu_num ${gpu_num}
    --exp_name ${exp_name}
    --test_subset ${test_subset}
    --n_average ${n_average}
    --beam_size ${beam_size}
    --max_tokens ${max_tokens}
    "

echo $cmd
eval $cmd
