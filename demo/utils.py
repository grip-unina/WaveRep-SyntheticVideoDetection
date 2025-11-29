import torch  # torch>=2.0.1
import timm   # timm>=1.0.12
import pandas
import numpy as np
from PIL import Image
import tqdm

def create_transform(cropping):
    import torchvision.transforms as transforms
    t = transforms.Compose([
        transforms.CenterCrop((cropping, cropping)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return t


def create_model(weights, arc, cropping, device):
    # create model 
    model = timm.create_model(arc, num_classes=1, pretrained=True, img_size=cropping)
    model = model.to(device)

    # load weights
    print('loading the model from %s' % weights)
    dat = torch.load(weights, map_location=device) 
    if 'state_dict' in dat:
        dat = {k[6:]: dat['state_dict'][k] for k in dat['state_dict'] if k.startswith('model')}
    model.load_state_dict(dat)
    del dat

    return model


class ReadVideoIteratorCV:
    def __init__(self, video_path, transform=None, limit=None):
        """
        Args:
            video_path (str): Path to video file.
            transform (callable, optional): Transform to apply to each frame.
        """
        self.video_path = video_path
        self.transform = transform
        self.limit = limit

    def __len__(self):
        import cv2
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if self.limit:
            total_frames = min(total_frames, self.limit)
        return total_frames

    def __iter__(self):
        import cv2
        cap = cv2.VideoCapture(self.video_path)
        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if self.transform:
                frame = self.transform(frame)
            yield idx, frame 
            idx = idx + 1
            if self.limit:
                if idx>=self.limit:
                    break
        cap.release()


class ReadVideoIteratorAV:
    def __init__(self, video_fid, transform=None, limit=None):
        """
        Args:
            video_fid (file id): video id file.
            transform (callable, optional): Transform to apply to each frame.
        """
        self.video_fid = video_fid
        self.transform = transform
        self.limit = limit

    def __len__(self):
        import av
        self.video_fid.seek(0)
        container = av.open(self.video_fid)
        total_frames = container.streams.video[0].frames
        container.close()
        if self.limit:
            total_frames = min(total_frames, self.limit)
        return total_frames

    def __iter__(self):
        import av
        self.video_fid.seek(0)
        container = av.open(self.video_fid)
        for idx, frame in enumerate(container.decode(video=0)):
            if self.limit:
                if idx>=self.limit:
                    break
            frame = frame.to_image().convert('RGB')
            if self.transform:
                frame = self.transform(frame)
            yield idx, frame
        container.close()


class BatchIterator:
    def __init__(self, source, batch_size):
        self.source = source
        self.batch_size = max(batch_size, 1)

    def __len__(self):
        len_source = len(self.source)
        if (len_source % self.batch_size) == 0:
            return len_source // self.batch_size 
        else:
            return len_source // self.batch_size + 1

    def __iter__(self):
        list_dat = list()
        list_idx = list()
        for idx, dat in self.source:
            list_dat.append(dat)
            list_idx.append(idx)
            if len(list_dat)>=self.batch_size:
                yield list_idx, torch.stack(list_dat, 0)
                list_dat = list()
                list_idx = list()
        if len(list_dat)>0:
            yield list_idx, torch.stack(list_dat, 0)


class ClipIterator:
    def __init__(self, reader, num_frames, stride=None):
        """
        Args:
            reader: reader video.
            num_frames (int): number of frames in a clip
            stride (int): stride between clips
        """
        self.reader = reader
        self.num_frames = max(num_frames, 1)
        self.stride = max(stride, 1) if stride else self.num_frames
        assert self.stride<=self.num_frames

    def __len__(self):
        len_video = len(self.reader)
        if len_video < self.num_frames:
            return 0
        else:
            return (len_video - self.num_frames) // self.stride + 1

    def __iter__(self):
        list_frame = list()
        list_idx   = list()
        for idx, frame in self.reader:
            list_frame.append(frame)
            list_idx.append(idx)
            if len(list_frame)>=self.num_frames:
                yield list_idx[len(list_idx)//2], torch.stack(list_frame, 1)
                list_frame = list_frame[self.stride:]
                list_idx = list_idx[self.stride:]


def evaluate(model, data_loader, device):
    list_out = list()
    with torch.no_grad():
        model = model.eval()
        pbar = tqdm.tqdm(data_loader)
        for idx, frame in pbar:
            pred = model(frame[None, ...].to(device))[0, -1].item()
            list_out.append({'index': idx, 'logit': pred})
            pbar.set_description('logit=%f' % pred)
    return pandas.DataFrame(list_out)


def evaluate_batch(model, data_loader, device):
    list_out = list()
    with torch.no_grad():
        model = model.eval()
        for idxs, frames in data_loader:
            preds = model(frames.to(device))[:, -1].cpu().numpy()
            for idx, pred in zip(idxs, preds):
                list_out.append({'index': idx, 'logit': pred})
    return pandas.DataFrame(list_out)

