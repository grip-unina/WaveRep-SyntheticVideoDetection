# download weights
echo -e "\n downloading weights"
mkdir -p "./weights"
wget -nc -P "./weights" "https://www.grip.unina.it/download/prog/WaveRep_SynthVideoDet/weights_dinov2_G4.ckpt" 

# download videos
echo -e "\n downloading videos"
mkdir -p "./videos"
wget -P "./videos" -nc "https://www.grip.unina.it/download/prog/WaveRep_SynthVideoDet/opensoraplan_-J24pksGK64_2.mp4"
wget -P "./videos" -nc "https://www.grip.unina.it/download/prog/WaveRep_SynthVideoDet/real_-D8sIkdnSkM_0.mp4"

# run on fake video
echo -e "\n running on fake video"
python main_avideo.py --video_input "./videos/opensoraplan_-J24pksGK64_2.mp4" --csv_output 'out_dinov2_G4_fake.csv'

# run on real video
echo -e "\n running on real video"
python main_avideo.py --video_input "./videos/real_-D8sIkdnSkM_0.mp4" --csv_output 'out_dinov2_G4_real.csv'

# run on a directory
#echo -e "\n running on video directory"
#python main_directory.py --vid_dir "./videos" --nameout dinov2_G4
#
