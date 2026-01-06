#!/bin/bash

: "${OUT_VIDEOS_DIR:=${1:-$(pwd)/videos}}"
: "${OUT_FRAMES_DIR:=${2:-$(pwd)/frames}}"

BROWSER=firefox

# extract cookies from browser
yt-dlp --cookies-from-browser "${BROWSER}" --cookies cookies.txt 2> /dev/null

for SET in 'test'
do

echo 'TODO' ${SET} real_panda
python script_download_real.py -i "./list_video_${SET}.csv" -v "${OUT_VIDEOS_DIR}/${SET}/real_panda" -f "${OUT_FRAMES_DIR}/${SET}/real_panda"
echo 'DONE' ${SET} real_panda

done
