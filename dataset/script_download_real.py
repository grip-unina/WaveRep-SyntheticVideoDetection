import os
import subprocess
import tqdm
import pandas
import yt_dlp

# To convert from the HH:MM:SS.MS to a float in second of the same timestamp
def convert_timestamp_to_second(timestamp):
    splits = timestamp.split(":")
    time_in_second = (float(splits[0])*60 + float(splits[1]))*60 + float(splits[2])
    return time_in_second

# Download video
def download_video(videoID, video_path):
        ydl_opts = {
            'format': 'mp4[height=720]',
            'merge_output_format': 'mp4',
            'outtmpl': os.path.splitext(video_path)[0]+'.%(ext)s',
            'cookiefile': './cookies.txt',
            'quiet': True,
            "no_warnings": True,
            'noplaylist': True
        }
        video_url = f"https://www.youtube.com/watch?v={videoID}"        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        

# Extract the frames 
def extract_frames(video_path, output_video_dir, start_time, end_time):
    os.makedirs(output_video_dir, exist_ok=True)

    subprocess.call(
        [
            "ffmpeg",
            "-loglevel","error",
            "-ss", str(start_time),
            "-to", str(end_time),
            "-i", video_path, "-y",
            "-start_number", "0",
            os.path.join(output_video_dir, "frame%04d.png"),
        ]
    )


def main(input_csv, output_videos_dir, output_frames_dir):
    print(f"downloading videos of {input_csv} to {output_videos_dir}")
    print(f"extracting  frames to {output_frames_dir}")
    os.makedirs(output_videos_dir, exist_ok=True)
    os.makedirs(output_frames_dir, exist_ok=True)
    table = pandas.read_csv(input_csv)
    
    pbar = tqdm.tqdm(table.index)
    for id in pbar:
        videoID = table.loc[id, 'videoID']
        index = table.loc[id, 'index']
        start_timestamp = convert_timestamp_to_second(table.loc[id, 'start_timestamp'])
        end_timestamp = convert_timestamp_to_second(table.loc[id, 'end_timestamp'])
        
        video_path = os.path.join(output_videos_dir, f'video_{videoID}_{index}.mp4')
        frame_path = os.path.join(output_frames_dir, f'video_{videoID}_{index}')
        if os.path.isfile(os.path.join(frame_path, "frame%04d.png"%64)): continue
        try:
            #if True:
            # Call function to download video
            pbar.set_description("downloading")
            download_video(videoID, video_path)

            # Call function to extract the frames with ffmpeg
            pbar.set_description("extracting ")
            extract_frames(video_path, frame_path, start_timestamp, end_timestamp)
        except yt_dlp.utils.DownloadError as e:
            print(e)
            if 'a bot' in e.msg:
                os.remove('./cookies.txt')
                print('Solutions:\n\t 1) log in to YouTube in your browser and then restart the script.\n\t 2) restart the script in a few hours.')
                exit()
        pbar.set_description("")

    print('DONE')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", '-i')
    parser.add_argument("--output_videos_dir", '-v')
    parser.add_argument("--output_frames_dir", '-f')
    args = parser.parse_args()
    main(args.input_csv, args.output_videos_dir, args.output_frames_dir)
