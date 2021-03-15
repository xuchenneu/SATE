#! /bin/bash

# training the model

gpu_num=0
update_freq=1

extra_tag=
extra_parameter=

#extra_tag="${extra_tag}"
#extra_parameter="${extra_parameter} "

exp_tag=test
train_config=asr_train_ctc.yaml

max_tokens=4000

cmd="./run.sh
    --stage 1
    --stop_stage 1
    --gpu_num ${gpu_num}
    --update_freq ${update_freq}
    --exp_tag ${exp_tag}
    --train_config ${train_config}
    --max_tokens ${max_tokens}
    "

if [[ -n ${extra_tag} ]]; then
    cmd="$cmd --extra_tag ${extra_tag}"
fi
if [[ -n ${extra_parameter} ]]; then
    cmd="$cmd --extra_parameter ${extra_parameter}"
fi

echo $cmd
eval $cmd
