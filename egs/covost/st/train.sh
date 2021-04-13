#! /bin/bash

# training the model

gpu_num=8
update_freq=2
max_tokens=20000

extra_tag=lcrm
extra_parameter=

#extra_tag="${extra_tag}"
#extra_parameter="${extra_parameter} "

exp_tag=baseline
train_config=train_ctc_sate.yaml
#train_config=train_ctc.yaml

cmd="./run.sh
    --stage 1
    --stop_stage 1
    --gpu_num ${gpu_num}
    --update_freq ${update_freq}
    --train_config ${train_config}
    --max_tokens ${max_tokens}
    "

if [[ -n ${exp_tag} ]]; then
    cmd="$cmd --exp_tag ${exp_tag}"
fi
if [[ -n ${extra_tag} ]]; then
    cmd="$cmd --extra_tag ${extra_tag}"
fi
if [[ -n ${extra_parameter} ]]; then
    cmd="$cmd --extra_parameter \"${extra_parameter}\""
fi

echo $cmd
eval $cmd
