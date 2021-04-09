gpu_num=1

while :
do
    all_devices=$(seq 0 `gpustat | sed '1,2d' | wc -l`);
    count=0
    for dev in ${all_devices[@]}
    do
        line=`expr $dev + 2`
        use=`gpustat -p | head -n $line | tail -1 | cut -d '|' -f4 | wc -w`
        if [[ $use -eq 0 ]]; then
            device[$count]=$dev
            count=`expr $count + 1`
            if [[ $count -eq $gpu_num ]]; then
                break
            fi
        fi
    done
    if [[ ${#device[@]} -lt $gpu_num ]]; then
        sleep 60s
    else
        echo "Run $cmd"
        eval $cmd
        sleep 10s
        exit
    fi
done
