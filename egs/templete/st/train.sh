#! /bin/bash

# training the model

gpu_num=8
update_freq=2
max_tokens=20000

exp_name=
extra_tag=
extra_parameter=

#extra_tag="${extra_tag}"
#extra_parameter="${extra_parameter} "

#extra_tag="${extra_tag}_encdlcl"
#extra_parameter="${extra_parameter} --use-enc-dlcl"

#extra_tag="${extra_tag}_decdlcl"
#extra_parameter="${extra_parameter} --use-dec-dlcl"

exp_tag=baseline
train_config=train_ctc.yaml
#train_config=train_ctc_conformer.yaml
#train_config=train_ctc_conformer_rpr.yaml
#train_config=train_ctc_sate.yaml
#train_config=train_ctc_sate_rpr.yaml
#train_config=train_ctc_sate_conformer.yaml
#train_config=train_ctc_sate_conformer_rpr.yaml

cmd="./run.sh
    --stage 1
    --stop_stage 1
    --gpu_num ${gpu_num}
    --update_freq ${update_freq}
    --train_config ${train_config}
    --max_tokens ${max_tokens}
    "

if [[ -n ${exp_name} ]]; then
    cmd="$cmd --exp_name ${exp_name}"
fi
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
