#!/bin/bash

: "${OUT_VIDEOS_DIR:=${1:-$(pwd)/videos}}"
: "${OUT_FRAMES_DIR:=${2:-$(pwd)/frames}}"

URL='https://www.grip.unina.it/download/prog/WaveRep_SynthVideoDet/datasets'

for SET in 'test'
do

for TYPE in 'pyramid_h264' 'opensoraplan_h264' 'cogvideox15_h264' 'allegro_h264' 'mochi1_h264' 'sora_web' 'flux_web' 'nove_h264'
do
    echo 'TODO' ${SET} ${TYPE}
    python script_download_fake.py -i "${URL}/${SET}__${TYPE}.zip" -v "${OUT_VIDEOS_DIR}/${SET}/${TYPE}" -f "${OUT_FRAMES_DIR}/${SET}/${TYPE}"
    echo 'DONE' ${SET} ${TYPE}
done

done
