# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Copyright (c) 2025 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
#
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt
#
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Example:
#     augmenter = waverep_d2('haar', prob=0.1, num_levels=3).eval()
#     frame0, frame1 = augmenter(frame0, frame1)
#

import torch
import torch.nn as nn
import pywt as numpy_wt
import random

class Troch_DWT(torch.nn.Module):
    def __init__(self, wavelet, padding_mode='zeros', axis=-1, num_levels=1):
        super().__init__()
        if isinstance(wavelet, str):
            if wavelet=='dif':
                wavelet = ([0.5,0.5],[-0.5,0.5])
            else:
                wavelet = numpy_wt.Wavelet(wavelet)
        
        if isinstance(wavelet, numpy_wt.Wavelet):
            f_l, f_h = wavelet.dec_lo, wavelet.dec_hi
        else:
            assert len(wavelet) == 2
            f_l, f_h = wavelet[0], wavelet[1]

        self.axis = axis
        self.num_levels = num_levels
        
        self.conv_l = torch.nn.Conv2d(1, 1, (len(f_l), 1), stride=(2,1), padding=((len(f_l)-1)//2,0), bias=False, padding_mode=padding_mode)
        self.conv_l.weight.data = torch.tensor(f_l, device=self.conv_l.weight.device, dtype=self.conv_l.weight.dtype)[None,None,:,None]
        
        self.conv_h = torch.nn.Conv2d(1, 1, (len(f_h), 1), stride=(2,1), padding=((len(f_h)-1)//2,0), bias=False, padding_mode=padding_mode)
        self.conv_h.weight.data = torch.tensor(f_h, device=self.conv_h.weight.device, dtype=self.conv_h.weight.dtype)[None,None,:,None]

    def forward(self, x, axis=None):
        x_shape = list(x.shape)
        axis = self.axis if axis is None else axis
        axis = axis % len(x_shape)
        x = x[None,...] if axis==0 else torch.flatten(x, start_dim=0, end_dim=axis-1)
        x = x[...,None] if len(x.shape)<3 else torch.flatten(x, start_dim=2)
        
        
        y = [None,] * (self.num_levels+1)
        for index in range(self.num_levels,0,-1):
            h = self.conv_h(x[:,None,:,:])[:,0,:,:]
            x = self.conv_l(x[:,None,:,:])[:,0,:,:]
            
            x_shape[axis] = h.shape[1]
            y[index] = h.view(*x_shape)
            
        x_shape[axis] = x.shape[1]
        y[0] = x.view(*x_shape)
        
        return tuple(y)


class Troch_iDWT(nn.Module):
    def __init__(self, wavelet, padding_mode='zeros', axis=-1):
        super().__init__()
        if isinstance(wavelet, str):
            wavelet = numpy_wt.Wavelet(wavelet)
        
        if isinstance(wavelet, numpy_wt.Wavelet):
            f_l, f_h = wavelet.dec_lo, wavelet.dec_hi
        else:
            assert len(wavelet) == 2
            f_l, f_h = wavelet[0], wavelet[1]

        self.axis = axis
        
        self.conv_l = torch.nn.ConvTranspose2d(1, 1, (len(f_l), 1), stride=(2,1), padding=((len(f_l)-1)//2,0), bias=False, padding_mode=padding_mode)
        self.conv_l.weight.data = torch.tensor(f_l, device=self.conv_l.weight.device, dtype=self.conv_l.weight.dtype)[None,None,:,None]
        
        self.conv_h = torch.nn.ConvTranspose2d(1, 1, (len(f_h), 1), stride=(2,1), padding=((len(f_h)-1)//2,0), bias=False, padding_mode=padding_mode)
        self.conv_h.weight.data = torch.tensor(f_h, device=self.conv_h.weight.device, dtype=self.conv_h.weight.dtype)[None,None,:,None]
        
    def forward(self, *x, axis=None):
        y = x[0]
        x_shape = list(y.shape)
        axis = self.axis if axis is None else axis
        axis = axis % len(x_shape)
        
        y = y[None,...] if axis==0 else torch.flatten(y, start_dim=0, end_dim=axis-1)
        y = y[...,None] if len(y.shape)<3 else torch.flatten(y, start_dim=2)
        
        for index in range(1,len(x)):
            x_h = x[index]
            x_shape[axis] += x_h.shape[axis]
            x_h = x_h[None,...] if axis==0 else torch.flatten(x_h, start_dim=0, end_dim=axis-1)
            x_h = x_h[...,None] if len(x_h.shape)<3 else torch.flatten(x_h, start_dim=2)
            y = self.conv_l(y[:,None,:,:])[:,0,:,:] + self.conv_h(x_h[:,None,:,:])[:,0,:,:]

        return y.view(*x_shape)
        

class waverep_d2(torch.nn.Module):
    def __init__(self, wavelet='haar', prob=0.1, num_levels=3):
        super().__init__()
        self.dwt = Troch_DWT(wavelet, num_levels=num_levels)
        self.idwt = Troch_iDWT(wavelet)
        self.prob = prob
        for p in self.parameters():
            p.requires_grad = False

    def _fun_dir(self, x):
        return [list(self.dwt(_, axis=-1)) for _ in self.dwt(x, axis=-2)]
    
    def _fun_inv(self, x):
        x = [self.idwt(*_, axis=-1) for _ in x]
        return self.idwt(*x, axis=-2)
        
    def forward(self, img0, img1):
        with torch.no_grad():
            img0T = self._fun_dir(img0)
            img1T = self._fun_dir(img1)
            if random.random()<self.prob:
                img1T[0][0] = img0T[0][0]
            if random.random()<self.prob:
                for p in range(1, len(img1T)):
                    img1T[0][p] = img0T[0][p]
                    img1T[p][0] = img0T[p][0]
            img1 = self._fun_inv(img1T)
        return img0, img1
    
    def pair(self, img0, img1):
        img0, img1 = self(img0, img1)
        y = torch.cat((img0, img1), 0)
        t = torch.cat((torch.zeros(len(img0), device=y.device),
                       torch.ones( len(img1), device=y.device)), 0)
        return y, t

