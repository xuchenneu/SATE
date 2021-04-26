set -e

gpu_num=1
root_dir=/home/xuchen/st/Fairseq-S2T
ckpt=/home/xuchen/st/checkpoints/mustc-v2/st

model_txt=$1
set=$2
test_subset=$3

#data_dir=/home/xuchen/st/data/mustc-v2/st_lcrm/en-de
#test_subset=(tst-COMMON)

data_dir=/media/data/tst/$set/en-de
#test_subset=(office)
#test_subset=(webrtc1)
#test_subset=(adap2)

data_config=config_st_share.yaml
result_file=./result

beam_size=5
lenpen=0.6
max_tokens=10000

models=()
i=0
for line in `cat $model_txt`; do
    i=`expr $i + 1`
    
    model_dir=$ckpt/$line
    [[ ! -d $model_dir ]] && echo $model_dir && exit 1;

    if [[ -f $model_dir/avg_10_checkpoint.pt ]]; then
        model=$model_dir/avg_10_checkpoint.pt
    else
        model=$model_dir/checkpoint_best.pt
    fi
    [[ ! -f $model ]] && echo $model && exit 1;

    models[$i]=$model
done

models=`echo ${models[*]} | sed 's/ /:/g'`

res_dir=$ckpt/ensemble/$set
i=0
while : 
do
    if [[ -d $res_dir/$i ]]; then
        i=`expr $i + 1`
    else
        res_dir=$res_dir/$i
        break
    fi 
done

mkdir -p $res_dir
cp $model_txt $res_dir


if [[ -z ${device} || ${#device[@]} -eq 0 ]]; then
    if [[ ${gpu_num} -eq 0 ]]; then
        device=()
    else
        source ./local/utils.sh
        device=$(get_devices $gpu_num 0)
    fi
fi
export CUDA_VISIBLE_DEVICES=${device}

for subset in ${test_subset[@]}; do
    subset=${subset}_st
    cmd="python ${root_dir}/fairseq_cli/generate.py
    ${data_dir}
    --config-yaml ${data_config}
    --gen-subset ${subset}
    --task speech_to_text
    --path ${models}
    --results-path ${res_dir}
    --skip-invalid-size-inputs-valid-test
    --max-tokens ${max_tokens}
    --beam ${beam_size}
    --lenpen ${lenpen}
    --scoring sacrebleu"
    echo -e "\033[34mRun command: \n${cmd} \033[0m"

    eval $cmd
    tail -n 1 ${res_dir}/generate-${subset}.txt

    cd $res_dir
    evaluate.sh translation-${subset}.txt $set
    cd -
done

