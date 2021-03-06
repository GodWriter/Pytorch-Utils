import os
import tqdm
import datetime
import argparse

import torch
import numpy as np
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


def get_wav(in_channels, pool=True):
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LL, LH, HL, HH


class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)


class WaveUnpool(nn.Module):
    def __init__(self, in_channels, option_unpool='cat5'):
        super(WaveUnpool, self).__init__()
        self.in_channels = in_channels
        self.option_unpool = option_unpool
        self.LL, self.LH, self.HL, self.HH = get_wav(self.in_channels, pool=False)

    def forward(self, LL, LH, HL, HH, original=None):
        if self.option_unpool == 'sum':
            return self.LL(LL) + self.LH(LH) + self.HL(HL) + self.HH(HH)
        elif self.option_unpool == 'cat5' and original is not None:
            return torch.cat([self.LL(LL), self.LH(LH), self.HL(HL), self.HH(HH), original], dim=1)
        else:
            raise NotImplementedError


class WaveEncoder(nn.Module):
    def __init__(self, option_unpool):
        super(WaveEncoder, self).__init__()
        self.option_unpool = option_unpool

        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)

        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.pool1 = WavePool(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.pool2 = WavePool(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.pool3 = WavePool(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 0)

    def forward(self, x):
        skips = {}
        for level in [1, 2, 3, 4]:
            x = self.encode(x, skips, level)
        return x

    def encode(self, x, skips, level):
        assert level in {1, 2, 3, 4}
        if self.option_unpool == 'sum':
            if level == 1:
                # print("1 Model.py: ", x.shape)
                out = self.conv0(x)
                out = self.relu(self.conv1_1(self.pad(out)))
                out = self.relu(self.conv1_2(self.pad(out)))
                skips['conv1_2'] = out
                LL, LH, HL, HH = self.pool1(out)
                skips['pool1'] = [LH, HL, HH]
                return LL
            elif level == 2:
                # print("2 Model.py: ", x.shape)
                out = self.relu(self.conv2_1(self.pad(x)))
                out = self.relu(self.conv2_2(self.pad(out)))
                skips['conv2_2'] = out
                LL, LH, HL, HH = self.pool2(out)
                skips['pool2'] = [LH, HL, HH]
                return LL
            elif level == 3:
                # print("3 Model.py: ", x.shape)
                out = self.relu(self.conv3_1(self.pad(x)))
                out = self.relu(self.conv3_2(self.pad(out)))
                out = self.relu(self.conv3_3(self.pad(out)))
                out = self.relu(self.conv3_4(self.pad(out)))
                skips['conv3_4'] = out
                LL, LH, HL, HH = self.pool3(out)
                skips['pool3'] = [LH, HL, HH]
                return LL
            else:
                # print("4 Model.py: ", x.shape)
                return self.relu(self.conv4_1(self.pad(x)))

        elif self.option_unpool == 'cat5':
            if level == 1:
                # print("1 Model.py: ", x.shape)
                out = self.conv0(x)
                out = self.relu(self.conv1_1(self.pad(out)))
                return out

            elif level == 2:
                # print("2 Model.py: ", x.shape)
                out = self.relu(self.conv1_2(self.pad(x)))
                skips['conv1_2'] = out
                LL, LH, HL, HH = self.pool1(out)
                skips['pool1'] = [LH, HL, HH]
                out = self.relu(self.conv2_1(self.pad(LL)))
                return out

            elif level == 3:
                # print("3 Model.py: ", x.shape)
                out = self.relu(self.conv2_2(self.pad(x)))
                skips['conv2_2'] = out
                LL, LH, HL, HH = self.pool2(out)
                skips['pool2'] = [LH, HL, HH]
                out = self.relu(self.conv3_1(self.pad(LL)))
                return out

            else:
                # print("4 Model.py: ", x.shape)
                out = self.relu(self.conv3_2(self.pad(x)))
                out = self.relu(self.conv3_3(self.pad(out)))
                out = self.relu(self.conv3_4(self.pad(out)))
                skips['conv3_4'] = out
                LL, LH, HL, HH = self.pool3(out)
                skips['pool3'] = [LH, HL, HH]
                out = self.relu(self.conv4_1(self.pad(LL)))
                return out
        else:
            raise NotImplementedError


class WaveDecoder(nn.Module):
    def __init__(self, option_unpool):
        super(WaveDecoder, self).__init__()
        self.option_unpool = option_unpool

        if option_unpool == 'sum':
            multiply_in = 1
        elif option_unpool == 'cat5':
            multiply_in = 5
        else:
            raise NotImplementedError

        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.conv4_1 = nn.Conv2d(512, 256, 3, 1, 0)

        self.recon_block3 = WaveUnpool(256, option_unpool)
        if option_unpool == 'sum':
            self.conv3_4 = nn.Conv2d(256*multiply_in, 256, 3, 1, 0)
        else:
            self.conv3_4_2 = nn.Conv2d(256*multiply_in, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_1 = nn.Conv2d(256, 128, 3, 1, 0)

        self.recon_block2 = WaveUnpool(128, option_unpool)
        if option_unpool == 'sum':
            self.conv2_2 = nn.Conv2d(128*multiply_in, 128, 3, 1, 0)
        else:
            self.conv2_2_2 = nn.Conv2d(128*multiply_in, 128, 3, 1, 0)
        self.conv2_1 = nn.Conv2d(128, 64, 3, 1, 0)

        self.recon_block1 = WaveUnpool(64, option_unpool)
        if option_unpool == 'sum':
            self.conv1_2 = nn.Conv2d(64*multiply_in, 64, 3, 1, 0)
        else:
            self.conv1_2_2 = nn.Conv2d(64*multiply_in, 64, 3, 1, 0)
        self.conv1_1 = nn.Conv2d(64, 3, 3, 1, 0)

    def forward(self, x, skips):
        for level in [4, 3, 2, 1]:
            x = self.decode(x, skips, level)
        return x

    def decode(self, x, skips, level):
        assert level in {4, 3, 2, 1}
        if level == 4:
            out = self.relu(self.conv4_1(self.pad(x)))
            LH, HL, HH = skips['pool3']
            original = skips['conv3_4'] if 'conv3_4' in skips.keys() else None
            out = self.recon_block3(out, LH, HL, HH, original)
            _conv3_4 = self.conv3_4 if self.option_unpool == 'sum' else self.conv3_4_2
            out = self.relu(_conv3_4(self.pad(out)))
            out = self.relu(self.conv3_3(self.pad(out)))
            return self.relu(self.conv3_2(self.pad(out)))
        elif level == 3:
            out = self.relu(self.conv3_1(self.pad(x)))
            LH, HL, HH = skips['pool2']
            original = skips['conv2_2'] if 'conv2_2' in skips.keys() else None
            out = self.recon_block2(out, LH, HL, HH, original)
            _conv2_2 = self.conv2_2 if self.option_unpool == 'sum' else self.conv2_2_2
            return self.relu(_conv2_2(self.pad(out)))
        elif level == 2:
            out = self.relu(self.conv2_1(self.pad(x)))
            LH, HL, HH = skips['pool1']
            original = skips['conv1_2'] if 'conv1_2' in skips.keys() else None
            out = self.recon_block1(out, LH, HL, HH, original)
            _conv1_2 = self.conv1_2 if self.option_unpool == 'sum' else self.conv1_2_2
            return self.relu(_conv1_2(self.pad(out)))
        else:
            return self.conv1_1(self.pad(x))


def wct_core(cont_feat, sty_feat, weight=0.2):
    # print("cont_feat: ", cont_feat.size())
    # print("sty_feat: ", sty_feat.size())
    # print("weight: ", weight.size())
    # print("weight: ", weight)

    targetFeature = (1.0 - weight) * cont_feat + weight * sty_feat
    return targetFeature


def feature_wct(content_feat, style_feat, alpha, weight):
    target_feature = wct_core(content_feat, style_feat, weight)

    target_feature = target_feature.view_as(content_feat)
    target_feature = alpha * target_feature + (1 - alpha) * content_feat # Note that the default alpha is 1.

    return target_feature


class WCT2:
    def __init__(self, 
                 model_path='./checkpoints', 
                 transfer_at=['encoder', 'skip', 'decoder'], 
                 option_unpool='cat5', 
                 device='cuda:0', 
                 verbose=False):

        self.transfer_at = set(transfer_at)
        assert not(self.transfer_at - set(['encoder', 'decoder', 'skip'])), 'invalid transfer_at: {}'.format(transfer_at)
        assert self.transfer_at, 'empty transfer_at'

        self.device = torch.device(device)
        self.verbose = verbose
        self.encoder = WaveEncoder(option_unpool).to(self.device)
        self.decoder = WaveDecoder(option_unpool).to(self.device)
        self.encoder.load_state_dict(torch.load(os.path.join(model_path, 'wave_encoder_{}_l4.pth'.format(option_unpool)), map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(torch.load(os.path.join(model_path, 'wave_decoder_{}_l4.pth'.format(option_unpool)), map_location=lambda storage, loc: storage))

    def encode(self, x, skips, level):
        return self.encoder.encode(x, skips, level)

    def decode(self, x, skips, level):
        return self.decoder.decode(x, skips, level)

    def get_all_feature(self, x):
        skips = {}
        feats = {'encoder': {}, 'decoder': {}}
        for level in [1, 2, 3, 4]:
            x = self.encode(x, skips, level)
            if 'encoder' in self.transfer_at:
                feats['encoder'][level] = x

        if 'encoder' not in self.transfer_at:
            feats['decoder'][4] = x
        for level in [4, 3, 2]:
            x = self.decode(x, skips, level)
            if 'decoder' in self.transfer_at:
                feats['decoder'][level - 1] = x
        return feats, skips

    def transfer(self, content, style, alpha=1, weight=0.3):
        content_feat, content_skips = content, {}
        style_feats, style_skips = self.get_all_feature(style)

        wct2_enc_level = [1, 2, 3, 4]
        wct2_dec_level = [1, 2, 3, 4]
        wct2_skip_level = ['pool1', 'pool2', 'pool3']

        for level in [1, 2, 3, 4]:
            content_feat = self.encode(content_feat, content_skips, level)
            if 'encoder' in self.transfer_at and level in wct2_enc_level:
                content_feat = feature_wct(content_feat, 
                                           style_feats['encoder'][level],
                                           alpha=alpha,
                                           weight=weight)

        if 'skip' in self.transfer_at:
            for skip_level in wct2_skip_level:
                for component in [0, 1, 2]:  # component: [LH, HL, HH]
                    content_skips[skip_level][component] = feature_wct(content_skips[skip_level][component], 
                                                                       style_skips[skip_level][component],
                                                                       alpha=alpha,
                                                                       weight=weight)

        for level in [4, 3, 2, 1]:
            if 'decoder' in self.transfer_at and level in style_feats['decoder'] and level in wct2_dec_level:
                content_feat = feature_wct(content_feat, 
                                           style_feats['decoder'][level],
                                           alpha=alpha,
                                           weight=weight)
            content_feat = self.decode(content_feat, content_skips, level)

        return content_feat

