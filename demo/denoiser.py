import os
import numpy as np
import math
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view


def make_index_along_axis(idx, axis, ndims):
    axis = axis % ndims        
    return tuple([idx if dim==axis else slice(None) for dim in range(ndims)])


def rescale_area(x, siz_out, axis=0):
    siz_in = x.shape[axis]
    ndims = len(x.shape)
    siz_out = int(siz_out)
    
    ## avvio
    price  = 2*siz_out
    budget = 2*siz_in 
    carry  = siz_in + siz_out
    index_in = 0
    while carry>price:
        index_in = index_in - 1
        carry = carry - price
    
    ## ciclo
    y_shape = list(x.shape); y_shape[axis] = siz_out
    e_shape = list(x.shape); del e_shape[axis]
    y = np.empty(y_shape, dtype=x.dtype)
    
    last_x = x[make_index_along_axis(index_in%siz_in, axis=axis, ndims=ndims)]
    
    for index_out in range(siz_out):
        wallet = budget
        e = np.zeros(e_shape, dtype=x.dtype)
        while wallet>0:
            if wallet>=carry:
                e = e + carry * last_x
                wallet = wallet - carry
                #print('I', index_out, index_in, carry, wallet)
                index_in = index_in + 1
                last_x = x[make_index_along_axis(index_in%siz_in, axis=axis, ndims=ndims)]
                carry = price
            else:
                e = e + wallet * last_x
                carry = carry - wallet
                #print('I', index_out, index_in, wallet, carry)
                wallet = 0
                break
        y[make_index_along_axis(index_out, axis=axis, ndims=ndims)] = e/budget
    
    return y


class get_denoiser:
    def __init__(self, sigma, device, padding=False):
        import torch
        num_levels = 17

        out_channel = 3
        network = make_net(3, kernels=[3, ] * num_levels,
                           features=[64, ] * (num_levels - 1) + [out_channel],
                           bns=[False, ] + [True, ] *
                           (num_levels - 2) + [False, ],
                           acts=['relu', ] * (num_levels - 1) + ['linear', ],
                           dilats=[1, ] * num_levels,
                           bn_momentum=0.1, padding=1 if padding else 0)
        self.sigma = sigma
        self.padding = padding
        if sigma == 1:
            weights_path = os.path.dirname(__file__)+"/zang_1.th"
        elif sigma == 0.1:
            weights_path = os.path.dirname(__file__)+"/zang_0.1.th"
        elif sigma == 5:
            weights_path = os.path.dirname(__file__)+"/zang_5.th"
        elif sigma == 7:
            weights_path = os.path.dirname(__file__)+"/zang_7.th"
        else:
            print("Sigma should be one")
            assert False
        state_dict = torch.load(weights_path, torch.device('cpu'))
        network.load_state_dict(state_dict["network"])
        self.device = device
        self.network = network.to(self.device).eval()

        print(weights_path)

    def __call__(self, img):
        import torch
        with torch.no_grad():
            img = torch.from_numpy(np.float32(img)).permute(2, 0, 1)[None, ...]
            res = self.network(img.to(self.device))[
                0].permute(1, 2, 0).cpu().numpy()
        return res

    def batch(self, list_img):
        import torch
        with torch.no_grad():
            if torch.is_tensor(list_img):
                x = list_img
            else:
                x = [torch.from_numpy(np.float32(img)).permute(2, 0, 1) for img in list_img]
                x = torch.stack(x, 0)
            res = self.network(x.to(self.device)).permute(0, 2, 3, 1).cpu().numpy()
        return res

    def denoise(self, img):
        import torch
        with torch.no_grad():
            img = torch.from_numpy(np.float32(img)).permute(2, 0, 1)[None, ...]
            noise = self.sigma / 256.0 * self.network(img.to(self.device)).cpu()
            if self.padding:
                img = img - noise
            else:
                img = img[:, :, 17:-17, 17:-17] - noise
            img = img[0].permute(1, 2, 0).numpy()
        return img

def conv_with_padding(in_planes, out_planes, kernelsize, stride=1, dilation=1, bias=False, padding = None):
    if padding is None:
        padding = kernelsize//2
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernelsize, stride=stride, dilation=dilation, padding=padding, bias=bias)

def conv_init(conv, act='linear'):
    r"""
    Reproduces conv initialization from DnCNN
    """
    n = conv.kernel_size[0] * conv.kernel_size[1] * conv.out_channels
    conv.weight.data.normal_(0, math.sqrt(2. / n))

def batchnorm_init(m, kernelsize=3):
    r"""
    Reproduces batchnorm initialization from DnCNN
    """
    n = kernelsize**2 * m.num_features
    m.weight.data.normal_(0, math.sqrt(2. / (n)))
    m.bias.data.zero_()

def make_activation(act):
    if act is None:
        return None
    elif act == 'relu':
        return nn.ReLU(inplace=True)
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'leaky_relu':
        return nn.LeakyReLU(inplace=True)
    elif act == 'softmax':
        return nn.Softmax()
    elif act == 'linear':
        return None
    else:
        assert(False)

def make_net(nplanes_in, kernels, features, bns, acts, dilats, bn_momentum = 0.1, padding=None):
    r"""
    :param nplanes_in: number of of input feature channels
    :param kernels: list of kernel size for convolution layers
    :param features: list of hidden layer feature channels
    :param bns: list of whether to add batchnorm layers
    :param acts: list of activations
    :param dilats: list of dilation factors
    :param bn_momentum: momentum of batchnorm
    :param padding: integer for padding (None for same padding)
    """

    depth = len(features)
    assert(len(features)==len(kernels))

    layers = list()
    for i in range(0,depth):
        if i==0:
            in_feats = nplanes_in
        else:
            in_feats = features[i-1]

        elem = conv_with_padding(in_feats, features[i], kernelsize=kernels[i], dilation=dilats[i], padding=padding, bias=not(bns[i]))
        conv_init(elem, act=acts[i])
        layers.append(elem)

        if bns[i]:
            elem = nn.BatchNorm2d(features[i], momentum = bn_momentum)
            batchnorm_init(elem, kernelsize=kernels[i])
            layers.append(elem)

        elem = make_activation(acts[i])
        if elem is not None:
            layers.append(elem)

    return nn.Sequential(*layers)
