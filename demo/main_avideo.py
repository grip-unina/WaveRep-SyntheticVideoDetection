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
    parser.add_argument('--video_input', type=str)
    parser.add_argument('--csv_output', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--limit', type=int, default=None)

    opt = parser.parse_args()
    device = torch.device(opt.device)
    transform = create_transform(cropping)
    model = create_model(opt.weights, arc, cropping, device)

    # prediction
    data_loader = ReadVideoIteratorCV(opt.video_input, transform=transform, limit=opt.limit)
    tab = evaluate(model, data_loader, device)

    # result and save
    score_logit = tab['logit'].mean()
    score_prob = expit(score_logit)
    print('result: logit=', score_logit,'; prob=', score_prob)
    if opt.csv_output:
        os.makedirs(os.path.dirname(os.path.join('.', opt.csv_output)), exist_ok=True)
        tab.to_csv(opt.csv_output, index=False)
    
