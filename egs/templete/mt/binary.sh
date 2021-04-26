set -e

eval=1

root_dir=~/st/Fairseq-S2T
data_dir=/home/xuchen/st/data/wmt/test
vocab_dir=/home/xuchen/st/data/wmt/mt/en-de/unigram32000_share
src_vocab_prefix=spm_unigram32000_share
tgt_vocab_prefix=spm_unigram32000_share

src_lang=en
tgt_lang=de
tokenize=1
splits=(newstest2014 newstest2016)

for split in ${splits[@]}; do
    src_file=${data_dir}/${split}.${src_lang}
    tgt_file=${data_dir}/${split}.${tgt_lang}

    if [[ ${tokenize} -eq 1 ]]; then
        cmd="tokenizer.perl -l ${src_lang} --threads 8 -no-escape < ${src_file} > ${src_file}.tok"
        echo -e "\033[34mRun command: \n${cmd} \033[0m"
        [[ $eval -eq 1 ]] && eval ${cmd}

        cmd="tokenizer.perl -l ${tgt_lang} --threads 8 -no-escape < ${tgt_file} > ${tgt_file}.tok"
        echo -e "\033[34mRun command: \n${cmd} \033[0m"
        [[ $eval -eq 1 ]] && eval ${cmd}
        src_file=${src_file}.tok
        tgt_file=${tgt_file}.tok
    fi

    cmd="cat ${src_file}"
    if [[ ${lcrm} -eq 1 ]]; then
        cmd="python local/lower_rm.py ${src_file}"
    fi
    cmd="${cmd}
    | spm_encode --model ${vocab_dir}/${src_vocab_prefix}.model
    --output_format=piece
    > ${src_file}.spm"

    echo -e "\033[34mRun command: \n${cmd} \033[0m"
    [[ $eval -eq 1 ]] && eval ${cmd}

    cmd="spm_encode
    --model ${vocab_dir}/${tgt_vocab_prefix}.model
    --output_format=piece
    < ${tgt_file}
    > ${tgt_file}.spm"
    echo -e "\033[34mRun command: \n${cmd} \033[0m"
    [[ $eval -eq 1 ]] && eval ${cmd}

    src_file=${src_file}.spm
    tgt_file=${tgt_file}.spm

    mkdir -p ${data_dir}/final
    cmd="cp ${src_file} ${data_dir}/final/${split}.${src_lang}"
    echo -e "\033[34mRun command: \n${cmd} \033[0m"
    [[ $eval -eq 1 ]] && eval ${cmd}

    cmd="cp ${tgt_file} ${data_dir}/final/${split}.${tgt_lang}"
    echo -e "\033[34mRun command: \n${cmd} \033[0m"
    [[ $eval -eq 1 ]] && eval ${cmd}
done

n_set=${#splits[*]}
for ((i=0;i<$n_set;i++)); do
    dataset[$i]=${data_dir}/final/${splits[$i]}
done
pref=`echo ${dataset[*]} | sed 's/ /,/g'`

cmd="python ${root_dir}/fairseq_cli/preprocess.py
    --source-lang ${src_lang}
    --target-lang ${tgt_lang}
    --testpref ${pref}
    --destdir ${data_dir}/data-bin
    --srcdict ${vocab_dir}/${src_vocab_prefix}.txt
    --tgtdict ${vocab_dir}/${tgt_vocab_prefix}.txt
    --workers 64"

echo -e "\033[34mRun command: \n${cmd} \033[0m"
[[ $eval -eq 1 ]] && eval ${cmd}