import os
import numpy as np
from tqdm import tqdm
from denoiser import rescale_area
from denoiser import get_denoiser
from PIL import Image
import torch
from utils import ReadVideoIteratorCV, BatchIterator
from matplotlib import colormaps

siz = (64, 224, 224)
out_shape = (224, 224)


def res2img(res_fft, out_shape=None, factor=4, cmap="inferno"):
    energy2 = np.mean(res_fft)
    res_fft = np.mean(res_fft / factor / energy2, -1).clip(0, 1)
    res_fft = np.fft.fftshift(res_fft)
    if out_shape:
        from cv2 import resize

        res_fft = resize(res_fft, out_shape)
    res_fft = colormaps[cmap](res_fft)
    return Image.fromarray(np.uint8(np.round(255 * res_fft)))


def transform(x):
    return torch.from_numpy(np.array(x, dtype=np.float32)).permute(2, 0, 1) / 256.0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--vid_dir", type=str)
    parser.add_argument("--npy_output", type=str, default="power_spectrum.npy")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--limit", type=int, default=1024)
    opt = parser.parse_args()

    folder = opt.vid_dir
    video_extensions = (".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv")

    video_names = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(video_extensions):
                video_names.append(os.path.relpath(os.path.join(root, file), folder))

    output_file = opt.npy_output
    if not output_file.endswith(".npy"):
        output_file = output_file + ".npy"
    fund = get_denoiser(1, torch.device(opt.device))

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    list_res_fft = list()
    # Iterating on videos
    for video_input in tqdm(video_names):
        data_loader = ReadVideoIteratorCV(
            os.path.join(folder, video_input), transform=transform, limit=opt.limit
        )
        data_loader = BatchIterator(data_loader, opt.batch_size)

        # Iterating on frames
        res = [fund.batch(frames) for index, frames in data_loader]
        res = np.concatenate(res, 0)
        res_fft = np.abs(np.fft.fftn(res, axes=(0, 1, 2))) ** 2  # 3d FFT
        res_fft = rescale_area(
            rescale_area(rescale_area(res_fft, siz[0], 0), siz[1], 1), siz[2], 2
        )  # resizing
        list_res_fft.append(res_fft)

    res_fft = np.mean(list_res_fft, 0)
    np.save(output_file, res_fft)

    out_png = output_file[:-4] + ".%s.png"
    res2img(res_fft.mean(0), out_shape=out_shape).save(out_png % "xy")
    res2img(res_fft.mean(1), out_shape=out_shape).save(out_png % "tx")
    res2img(np.swapaxes(res_fft.mean(2), 1, 0), out_shape=out_shape).save(
        out_png % "yt"
    )
