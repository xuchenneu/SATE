set -e

eval=1

lcrm=1
tokenizer=0

root_dir=~/st/Fairseq-S2T
data_dir=/home/xuchen/st/data/test
vocab_dir=/home/xuchen/st/data/mustc/st_lcrm/en-de
asr_vocab_prefix=spm_unigram10000_st_share
st_vocab_prefix=spm_unigram10000_st_share

src_lang=en
tgt_lang=de
splits=(2019)

source ~/tools/audio/bin/activate

splits=`echo ${splits[*]} | sed 's/ /,/g'`

cp -r ${vocab_dir}/${asr_vocab_prefix}.* ${data_dir}/${src_lang}-${tgt_lang}
cp -r ${vocab_dir}/${st_vocab_prefix}.* ${data_dir}/${src_lang}-${tgt_lang}
rm -rf ${data_dir}/${src_lang}-${tgt_lang}/fbank80.zip

cmd="python ${root_dir}/examples/speech_to_text/prep_st_data.py
    --data-root ${data_dir}
    --output-root ${data_dir}
    --splits ${splits}
    --task st
    --src-lang ${src_lang}
    --tgt-lang ${tgt_lang}
    --add-src
    --share
    --asr-prefix ${asr_vocab_prefix}
    --st-spm-prefix ${st_vocab_prefix}
    --cmvn-type utterance"

    if [[ ${lcrm} -eq 1 ]]; then
        cmd="$cmd
    --lowercase-src
    --rm-punc-src"
    fi
    if [[ ${tokenizer} -eq 1 ]]; then
        cmd="$cmd
    --tokenizer"
    fi

echo -e "\033[34mRun command: \n${cmd} \033[0m"
[[ $eval -eq 1 ]] && eval ${cmd}
deactivate
