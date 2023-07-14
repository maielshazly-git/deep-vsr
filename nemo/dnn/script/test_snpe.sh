#!/bin/bash

function _usage()
{
cat << EOF
_usage: $(basename ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}) [-c CONTENT] [-d DEVICE_ID] [-q QUALITY] [-r RESOLUTION] [-s SCALE]
EOF
}

function _set_conda(){
    # >>> conda initialize >>>
    # !! Contents within this block are managed by 'conda init' !!
    __conda_setup="$('/home/mai/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)" # '/opt/conda/bin/conda' My Fix
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "/home/mai/anaconda3/etc/profile.d/conda.sh" ]; then # My Fix
            . "/home/mai/anaconda3/etc/profile.d/conda.sh" # My Fix
        else
            export PATH="/home/mai/anaconda3/bin:$PATH" # My Fix
        fi
    fi
    unset __conda_setup
    # <<< conda initialize <<<
    conda deactivate
    conda activate nemo_py3.5
}

function _set_bitrate(){
    if [ "$1" == 240 ];then
        bitrate=512
    elif [ "$1" == 360 ];then
        bitrate=1024
    elif [ "$1" == 480 ];then
        bitrate=1600
    fi
}

function _set_num_blocks(){
    if [ "$1" == 240 ];then
        if [ "$2" == "low" ];then
            num_blocks=4
        elif [ "$2" == "medium" ];then
            num_blocks=8
        elif [ "$2" == "high" ];then
            num_blocks=8
        fi
    elif [ "$1" == 360 ];then
        if [ "$2" == "low" ];then
            num_blocks=4
        elif [ "$2" == "medium" ];then
            num_blocks=4
        elif [ "$2" == "high" ];then
            num_blocks=4
        fi
    elif [ "$1" == 480 ];then
        if [ "$2" == "low" ];then
            num_blocks=4
        elif [ "$2" == "medium" ];then
            num_blocks=4
        elif [ "$2" == "high" ];then
            num_blocks=4
        fi
    fi
}

function _set_num_filters(){
    if [ "$1" == 240 ];then
        if [ "$2" == "low" ];then
            num_filters=9
        elif [ "$2" == "medium" ];then
            num_filters=21
        elif [ "$2" == "high" ];then
            num_filters=32
        fi
    elif [ "$1" == 360 ];then
        if [ "$2" == "low" ];then
            num_filters=8
        elif [ "$2" == "medium" ];then
            num_filters=18
        elif [ "$2" == "high" ];then
            num_filters=29
        fi
    elif [ "$1" == 480 ];then
        if [ "$2" == "low" ];then
            num_filters=4
        elif [ "$2" == "medium" ];then
            num_filters=9
        elif [ "$2" == "high" ];then
            num_filters=18
        fi
    fi
}

[[ ($# -ge 1)  ]] || { echo "[ERROR] Invalid number of arguments. See -h for help."; exit 1;  }

while getopts ":c:q:r:d:s:h" opt; do
    case $opt in
        h) _usage; exit 0;;
        c) content=("$OPTARG");;
        q) quality=("$OPTARG");;
        r) resolution=("$OPTARG");;
        d) device_id=("$OPTARG");;
        s) scale=("$OPTARG");;
        \?) exit 1;
    esac
done

if [ -z "${content+x}" ]; then
    echo "[ERROR] contents is not set"
    exit 1;
fi

if [ -z "${device_id+x}" ]; then
    echo "[ERROR] device_id is not set"
    exit 1;
fi

if [ -z "${scale+x}" ]; then
    echo "[ERROR] scale is not set"
    exit 1;
fi

if [ -z "${quality+x}" ]; then
    echo "[ERROR] quality is not set"
    exit 1;
fi

if [ -z "${resolution+x}" ]; then
    echo "[ERROR] resolution is not set"
    exit 1;
fi

_set_conda
_set_bitrate ${resolution}
_set_num_blocks ${resolution} ${quality}
_set_num_filters ${resolution} ${quality}
CUDA_VISIBLE_DEVICES=0 python ${NEMO_CODE_ROOT}/nemo/dnn/test_snpe.py --data_dir ${NEMO_DATA_ROOT} --content ${content} --video_name ${resolution}p_${bitrate}kbps_s0_d300.webm  --num_blocks ${num_blocks} --num_filters ${num_filters} --train_type train_video --device_id ${device_id} --scale ${scale}
