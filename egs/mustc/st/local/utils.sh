
get_devices(){
    gpu_num=$1
    use_cpu=$2
    device=()
    while :
    do
        record=`mktemp -t temp.record.XXXXXX`
        gpustat > $record
        all_devices=$(seq 0 `cat $record | sed '1,2d' | wc -l`);
        count=0
        for dev in ${all_devices[@]}
        do
            line=`expr $dev + 2`
            use=`cat $record | head -n $line | tail -1 | cut -d '|' -f3 | cut -d '/' -f1`
            if [[ $use -lt 100 ]]; then
                device[$count]=$dev
                count=`expr $count + 1`
                if [[ $count -eq $gpu_num ]]; then
                    break
                fi
            fi
        done
        if [[ ${#device[@]} -lt $gpu_num ]]; then
            if [[ $use_cpu -eq 1 ]]; then
                device=(-1)
            else
                sleep 60s
            fi
        else
            break
        fi
    done

    echo ${device[*]} | sed 's/ /,/g'
    return $?
}


