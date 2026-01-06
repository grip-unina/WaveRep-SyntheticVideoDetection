import os
import argparse
import subprocess
from tqdm import tqdm


# Extract the frames 
def extract_frames(video_path, output_video_dir):
    os.makedirs(output_video_dir, exist_ok=True)

    subprocess.call(
        [
            "ffmpeg",
            "-loglevel","error",
            "-i", video_path,
            "-start_number", "0",
            os.path.join(output_video_dir, "frame%04d.png"),
        ]
    )


# Download zip 
def download_zip(zip_url, output_videos_dir):
    filename = zip_url.split('/')[-1]
    os.makedirs(output_videos_dir, exist_ok=True)
    zip_path = os.path.join(output_videos_dir, filename)
    subprocess.call(
        [
            "wget",
            "-nc",zip_url,
            "-O", zip_path,
        ]
    )
    subprocess.call(
        [
            "unzip",
            "-nqj", zip_path,
            "-d", output_videos_dir,
        ]
    )

# find videos
def find_videos(videos_dir):
    video_list = []
    for dirpath, dirnames, filenames in os.walk(videos_dir):
        for filename in filenames:
            if filename.endswith(".mp4") and "flat" not in dirpath:
                video_list.append(os.path.join(dirpath, filename))
    return video_list


def main(zip_url, output_videos_dir, output_frames_dir):
    print(f"Downloading from {zip_url} to {output_videos_dir}")
    download_zip(zip_url, output_videos_dir)

    print(f"Starting to extract frames from {output_videos_dir} to {output_frames_dir}")
    for video_path in tqdm(find_videos(output_videos_dir)):
        output_video_dir = os.path.relpath(video_path, output_videos_dir)
        output_video_dir = os.path.join(output_frames_dir, output_video_dir)
        output_video_dir = os.path.splitext(output_video_dir)[0]
        extract_frames(video_path, output_video_dir)
    
    print('DONE')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_url", '-i')
    parser.add_argument("--output_videos_dir", '-v')
    parser.add_argument("--output_frames_dir", '-f')
    args = parser.parse_args()
    main(args.zip_url, args.output_videos_dir, args.output_frames_dir)
