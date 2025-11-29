# WaveRep SyntheticVideoDetection - demo code
Before using the code, download the weights:

```
mkdir -p ./weights
wget -nc -P ./weights "https://www.grip.unina.it/download/prog/WaveRep_SynthVideoDet/weights_dinov2_G1.ckpt"
wget -nc -P ./weights "https://www.grip.unina.it/download/prog/WaveRep_SynthVideoDet/weights_dinov2_G4.ckpt" 

# md5sum weights_dinov2_G1.ckpt > bc9852a7c1d3bfdb60a20b4497e575bb
# md5sum weights_dinov2_G4.ckpt > 8bf19e6f68a92bed600dd97fbed3f2cd
```

The `main_avideo.py` script requires as input a video file and provides a CSV file with the LLR score for each frame. If LLR>0, the frame is detected as synthetic.
The `main_directory.py` script requires as input a folder with video files and provides a CSV file for each video.

Here is an example of how to execute the script with the model trained with four generators:

```
python main_avideo.py --video_input "<VIDEO_PATH>" --csv_output '<CSV_OUTPUT_FILE>' --weights './weights/weights_dinov2_G4.ckpt'
```

Example of using with the model trained with a single generators:

```
python main_avideo.py --video_input "<VIDEO_PATH>" --csv_output '<CSV_OUTPUT_FILE>' --weights './weights/weights_dinov2_G1.ckpt'
```

In order to use the scripts the follwing packages should be installed:

	* torch>=2.0.1
    * timm>=1.0.12
    * opencv-python>=4.8.1.78
    * torchvision
	* pillow
	* tqdm
	* pandas
	* scipy
    * numpy