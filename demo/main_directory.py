import torch  # torch>=2.0.1
import timm   # timm>=1.0.12   
import os
import tqdm
import glob
import pandas
from scipy.special import expit

PATH_WEIGHTS='./weights/weights_dinov2_G4.ckpt'
cropping=504
arc='vit_base_patch14_reg4_dinov2.lvd142m'

from utils import create_transform, create_model, ReadVideoIteratorCV, evaluate

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default=PATH_WEIGHTS,
        type=str,
        help="weights of network",
    )
    parser.add_argument('--vid_dir', type=str)
    parser.add_argument('--nameout', type=str)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--st', type=int, default=0)
    parser.add_argument('--en', type=int, default=None)
    parser.add_argument('--limit', type=int, default=None)
    opt = parser.parse_args()

    folder = opt.vid_dir
    video_extensions = ('.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv')

    video_names = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(video_extensions):
                video_names.append(os.path.relpath(os.path.join(root, file), folder))
    video_names = video_names[opt.st:opt.en]

    nameout = opt.nameout
    device = torch.device(opt.device)
    transform = create_transform(cropping)
    model = create_model(opt.weights, arc, cropping, device)

    csv_output = os.path.join(folder, f'summary_{nameout}.csv')

    # prediction
    total_tab = list()
    for video_input in video_names:
        print("doing video: ", video_input)
        video_csv_output = os.path.join(folder, os.path.splitext(video_input)[0] + f'.{nameout}.csv')
        if os.path.isfile(video_csv_output):
            tab = pandas.read_csv(video_csv_output)
        else:
            data_loader = ReadVideoIteratorCV(os.path.join(folder, video_input), transform=transform, limit=opt.limit)
            tab = evaluate(model, data_loader, device)
            tab.to_csv(video_csv_output, index=False)

        # result and save
        if opt.limit:
            score_logit = tab['logit'].iloc[:opt.limit].mean()
        else:
            score_logit = tab['logit'].mean()
        score_prob = expit(score_logit)
        print('result: logit=', score_logit,'; prob=', score_prob, '(closer to 1 = more likely video is synthetic)')
        total_tab.append({'filename': video_input, 'logit': score_logit, 'prob': score_prob})
        print(video_input + " done!")
        
    total_tab = pandas.DataFrame(total_tab)
    os.makedirs(os.path.dirname(csv_output), exist_ok=True)
    total_tab.to_csv(csv_output, index=False)
    print('\n summary:')
    print(total_tab.to_csv(sep='\t', index=False))
        
    
