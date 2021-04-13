#! /bin/bash

# Processing MuST-C Datasets

# Copyright 2021 Natural Language Processing Laboratory 
# Xu Chen (xuchenneu@163.com)

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
#set -u
set -o pipefail
export PYTHONIOENCODING=UTF-8

eval=1
time=$(date "+%m%d_%H%M")

stage=0
stop_stage=0

######## hardware ########
# devices
#device=()
gpu_num=8
update_freq=1

root_dir=~/st/Fairseq-S2T
pwd_dir=$PWD

# dataset
src_lang=en
tgt_lang=de
lang=${src_lang}-${tgt_lang}

dataset=mustc
task=translation
vocab_type=unigram
vocab_size=10000
share_dict=1
lc_rm=1

use_specific_dict=1
specific_prefix=st_share10k_lcrm
specific_dir=/home/xuchen/st/data/mustc/st_lcrm/en-de
src_vocab_prefix=spm_unigram10000_st_share
tgt_vocab_prefix=spm_unigram10000_st_share

org_data_dir=/media/data/${dataset}
data_dir=~/st/data/${dataset}/mt/${lang}
train_subset=train
valid_subset=dev
test_subset=tst-COMMON
trans_set=test

# exp
extra_tag=
extra_parameter=
exp_tag=baseline
exp_name=

# config
train_config=train.yaml

# training setting
fp16=1
max_tokens=4096
step_valid=0
bleu_valid=0

# decoding setting
n_average=10
beam_size=5

if [[ ${use_specific_dict} -eq 1 ]]; then
    exp_tag=${specific_prefix}_${exp_tag}
    data_dir=${data_dir}/${specific_prefix}
    mkdir -p ${data_dir}
else
    data_dir=${data_dir}/${vocab_type}${vocab_size}
    src_vocab_prefix=spm_${vocab_type}${vocab_size}_${src_lang}
    tgt_vocab_prefix=spm_${vocab_type}${vocab_size}_${tgt_lang}
    if [[ $share_dict -eq 1 ]]; then
        data_dir=${data_dir}_share
        src_vocab_prefix=spm_${vocab_type}${vocab_size}_share
        tgt_vocab_prefix=spm_${vocab_type}${vocab_size}_share
    fi
fi

. ./local/parse_options.sh || exit 1;

# full path
train_config=$pwd_dir/conf/${train_config}
if [[ -z ${exp_name} ]]; then
    exp_name=$(basename ${train_config%.*})_${exp_tag}
    if [[ -n ${extra_tag} ]]; then
        exp_name=${exp_name}_${extra_tag}
    fi
fi
model_dir=$root_dir/../checkpoints/$dataset/mt/${exp_name}

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    # pass
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    echo "stage 0: MT Data Preparation"
    if [[ ! -e ${data_dir} ]]; then
        mkdir -p ${data_dir}
    fi

    if [[ ! -f ${data_dir}/${src_vocab_prefix}.txt || ! -f ${data_dir}/${tgt_vocab_prefix}.txt ]]; then
        if [[ ${use_specific_dict} -eq 0 ]]; then
            cmd="python ${root_dir}/examples/speech_to_text/prep_mt_data.py
                --data-root ${org_data_dir}
                --output-root ${data_dir}
                --splits ${train_subset},${valid_subset},${test_subset}
                --src-lang ${src_lang}
                --tgt-lang ${tgt_lang}
                --vocab-type ${vocab_type}
                --vocab-size ${vocab_size}"
            if [[ $share_dict -eq 1 ]]; then
                cmd="$cmd
                --share"
            fi
            echo -e "\033[34mRun command: \n${cmd} \033[0m"
            [[ $eval -eq 1 ]] && eval ${cmd}
        else
            cp -r ${specific_dir}/${src_vocab_prefix}.* ${data_dir}
            cp ${specific_dir}/${tgt_vocab_prefix}.* ${data_dir}
        fi
    fi

    mkdir -p ${data_dir}/data
    for split in ${train_subset} ${valid_subset} ${test_subset}; do
    {
        cmd="cat ${org_data_dir}/${lang}/data/${split}.${src_lang}"
        if [[ ${lc_rm} -eq 1 ]]; then
            cmd="python local/lower_rm.py ${org_data_dir}/${lang}/data/${split}.${src_lang}"
        fi
        cmd="${cmd}
        | spm_encode --model ${data_dir}/${src_vocab_prefix}.model
        --output_format=piece
        > ${data_dir}/data/${split}.${src_lang}"

        echo -e "\033[34mRun command: \n${cmd} \033[0m"
        [[ $eval -eq 1 ]] && eval ${cmd}

        cmd="spm_encode
        --model ${data_dir}/${tgt_vocab_prefix}.model
        --output_format=piece
        < ${org_data_dir}/${lang}/data/${split}.${tgt_lang}
        > ${data_dir}/data/${split}.${tgt_lang}"

        echo -e "\033[34mRun command: \n${cmd} \033[0m"
        [[ $eval -eq 1 ]] && eval ${cmd}
    }&
    done
    wait

    cmd="python ${root_dir}/fairseq_cli/preprocess.py
        --source-lang ${src_lang} --target-lang ${tgt_lang}
        --trainpref ${data_dir}/data/${train_subset}
        --validpref ${data_dir}/data/${valid_subset}
        --testpref ${data_dir}/data/${test_subset}
        --destdir ${data_dir}/data-bin
        --srcdict ${data_dir}/${src_vocab_prefix}.txt
        --tgtdict ${data_dir}/${tgt_vocab_prefix}.txt
        --workers 64"

    echo -e "\033[34mRun command: \n${cmd} \033[0m"
    [[ $eval -eq 1 ]] && eval ${cmd}
fi

data_dir=${data_dir}/data-bin

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: MT Network Training"
    [[ ! -d ${data_dir} ]] && echo "The data dir ${data_dir} is not existing!" && exit 1;

    if [[ -z ${device} || ${#device[@]} -eq 0 ]]; then
		if [[ ${gpu_num} -eq 0 ]]; then
			device=()
		else
        	source ./local/utils.sh
        	device=$(get_devices $gpu_num 0)
		fi
    fi

    echo -e "dev=${device} data=${data_dir} model=${model_dir}"

    if [[ ! -d ${model_dir} ]]; then
        mkdir -p ${model_dir}
    else
        echo "${model_dir} exists."
    fi

    cp ${BASH_SOURCE[0]} ${model_dir}
    cp ${PWD}/train.sh ${model_dir}
    cp ${train_config} ${model_dir}

    cmd="python3 -u ${root_dir}/fairseq_cli/train.py
        ${data_dir}
        --source-lang ${src_lang}
        --target-lang ${tgt_lang}
        --train-config ${train_config}
        --task ${task}
        --max-tokens ${max_tokens}
        --update-freq ${update_freq}
        --log-interval 100
        --save-dir ${model_dir}
        --tensorboard-logdir ${model_dir}"

    if [[ -n ${extra_parameter} ]]; then
        cmd="${cmd}
        ${extra_parameter}"
    fi
	if [[ ${gpu_num} -gt 0 ]]; then
		cmd="${cmd}
        --distributed-world-size $gpu_num
        --ddp-backend no_c10d"
	fi
    if [[ $fp16 -eq 1 ]]; then
        cmd="${cmd}
        --fp16"
    fi
    if [[ $step_valid -eq 1 ]]; then
        validate_interval=1
        save_interval=1
        keep_last_epochs=10
        no_epoch_checkpoints=0
        save_interval_updates=10000
        keep_interval_updates=10
    else
        validate_interval=1
        keep_last_epochs=10
    fi
    if [[ $bleu_valid -eq 1 ]]; then
        cmd="$cmd
        --eval-bleu
        --eval-bleu-args '{\"beam\": 1}'
        --eval-tokenized-bleu
        --eval-bleu-remove-bpe
        --best-checkpoint-metric bleu
        --maximize-best-checkpoint-metric"
    fi
    if [[ -n $no_epoch_checkpoints && $no_epoch_checkpoints -eq 1 ]]; then
        cmd="$cmd
        --no-epoch-checkpoints"
    fi
    if [[ -n $validate_interval ]]; then
        cmd="${cmd}
        --validate-interval $validate_interval "
    fi
    if [[ -n $save_interval ]]; then
        cmd="${cmd}
        --save-interval $save_interval "
    fi
    if [[ -n $keep_last_epochs ]]; then
        cmd="${cmd}
        --keep-last-epochs $keep_last_epochs "
    fi
    if [[ -n $save_interval_updates ]]; then
        cmd="${cmd}
        --save-interval-updates $save_interval_updates"
        if [[ -n $keep_interval_updates ]]; then
        cmd="${cmd}
        --keep-interval-updates $keep_interval_updates"
        fi
    fi

    echo -e "\033[34mRun command: \n${cmd} \033[0m"

    # save info
    log=./history.log
    echo "${time} | ${device} | ${data_dir} | ${model_dir} " >> $log
    cat $log | tail -n 50 > tmp.log
    mv tmp.log $log
    export CUDA_VISIBLE_DEVICES=${device}

    cmd="nohup ${cmd} >> ${model_dir}/train.log 2>&1 &"
    if [[ $eval -eq 1 ]]; then
		eval $cmd
		sleep 2s
		tail -n `wc -l ${model_dir}/train.log | awk '{print $1+1}'` -f ${model_dir}/train.log
	fi
fi
wait

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: MT Decoding"
    if [[ ${n_average} -ne 1 ]]; then
        # Average models
		dec_model=avg_${n_average}_checkpoint.pt

		cmd="python ${root_dir}/scripts/average_checkpoints.py
        --inputs ${model_dir}
        --num-epoch-checkpoints ${n_average}
        --output ${model_dir}/${dec_model}"
    	echo -e "\033[34mRun command: \n${cmd} \033[0m"
    	[[ $eval -eq 1 ]] && eval $cmd
	else
		dec_model=checkpoint_best.pt
	fi

    if [[ -z ${device} || ${#device[@]} -eq 0 ]]; then
		if [[ ${gpu_num} -eq 0 ]]; then
			device=()
		else
        	source ./local/utils.sh
        	device=$(get_devices $gpu_num 0)
		fi
    fi
    export CUDA_VISIBLE_DEVICES=${device}

	#tmp_file=$(mktemp ${model_dir}/tmp-XXXXX)
	#trap 'rm -rf ${tmp_file}' EXIT
	result_file=${model_dir}/decode_result
	[[ -f ${result_file} ]] && rm ${result_file}

    trans_set=(${trans_set//,/ })
	for subset in ${trans_set[@]}; do
  		cmd="python ${root_dir}/fairseq_cli/generate.py
        ${data_dir}
        --source-lang ${src_lang}
        --target-lang ${tgt_lang}
        --gen-subset ${subset}
        --task ${task}
        --path ${model_dir}/${dec_model}
        --results-path ${model_dir}
        --max-tokens ${max_tokens}
        --beam ${beam_size}
        --post-process sentencepiece
        --tokenizer moses
        --moses-source-lang ${src_lang}
        --moses-target-lang ${tgt_lang}
        --scoring sacrebleu"
    	echo -e "\033[34mRun command: \n${cmd} \033[0m"

        if [[ $eval -eq 1 ]]; then
    	    eval $cmd
    	    tail -n 1 ${model_dir}/generate-${subset}.txt >> ${result_file}
        fi
	done
    cat ${result_file}
fi
