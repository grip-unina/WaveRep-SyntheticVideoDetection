### TestData download
To download the test set videos and extract their frames, run the following commands:
```
    export OUT_VIDEOS_DIR=./videos
    export OUT_FRAMES_DIR=./frames
    bash ./download_fake_test.sh
    bash ./download_real_test.sh
```

You can specify custom output directories for videos and frames by changing the environment variables `OUT_VIDEOS_DIR` and `OUT_FRAMES_DIR`.

It relies on two bash scripts, one for fake data and one for real data:
- Fake script: Downloads ZIP archives containing videos and extracts all frames from each video.
- Real script: Downloads videos from YouTube and extracts only the frames corresponding to the relevant clip segment for each video.

The scripts require the following dependencies: python, pandas, tqdm, yt-dlp, wget, unzip, firefox, ffmpeg.

### MD5 Checksums
Below are the MD5 checksums for verifying the integrity of the downloaded ZIP files:

```
3a9d579bc20fa1512dc7eee86ab6014b  test__allegro_h264.zip
f24e43ea934e0f44e2a4845079f937e8  test__cogvideox15_h264.zip
9acf6906705fda9d2dc81a14a4d6a2a1  test__flux_web.zip
59ac77901c3a374e153b102276fb5ff6  test__mochi1_h264.zip
bf068519e360586d07dee9216b273ae1  test__nove_h264.zip
0150fcab5d288376faec4aadf4a90e63  test__opensoraplan_h264.zip
8273f9cff90eb73cb21707ba5edf0d59  test__pyramid_h264.zip
72331fe9f00d2be575a160d25230b98d  test__sora_web.zip
```
