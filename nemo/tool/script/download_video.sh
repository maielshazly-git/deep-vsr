#!/bin/bash

function _usage()
{
cat << EOF
_usage: $(basename ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}) [-c CONTENTS]

mandatory arguments:
-c CONTENTS                 Specifies contents (e.g., product_review)

EOF
}

while getopts ":c:h" opt; do
    case $opt in
        h) _usage; exit 0;;
        c) contents+=("$OPTARG");;
        \?) exit 1;
    esac
done

if [ -z "${contents}" ] ; then
    echo "[ERROR] contents is not set"
    exit 1;
fi

for content in "${contents[@]}"
do
    python /home/mai/nemo/nemo/tool/download_video.py --video_dir /home/mai/nemo/data/nemo-data-traffic/video --content ${content}
done
