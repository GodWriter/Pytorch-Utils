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


IMG_EXTENSIONS = ['.jpg', '.JPG', '.png', '.PNG',]


class Timer:
    def __init__(self, msg='Elapsed time: {}', verbose=True):
        self.msg = msg
        self.start_time = None
        self.verbose = verbose

    def __enter__(self):
        self.start_time = datetime.datetime.now()

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.verbose:
            print(self.msg.format(datetime.datetime.now() - self.start_time))


def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
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


def wct_core_bbox(cont_feat, sty_feat, bbox, cont_weight=0.7, sty_weight=0.3, device='cpu'):
    _, _, H, W = cont_feat.shape
    bboxXXYY = bbox.detach().type(torch.cuda.IntTensor)

    bboxXXYY[:, 1] = (bbox[:, 1] - bbox[:, 3] / 2) * W
    bboxXXYY[:, 2] = (bbox[:, 2] - bbox[:, 4] / 2) * H
    bboxXXYY[:, 3] = (bbox[:, 1] + bbox[:, 3] / 2) * W
    bboxXXYY[:, 4] = (bbox[:, 2] + bbox[:, 4] / 2) * H

    contMask = torch.ones(H, W)
    contMask = contMask.to(device)

    styMask = torch.zeros(H, W)
    styMask = styMask.to(device)

    for box in bboxXXYY:
        xmin, ymin, xmax, ymax = box[1], box[2], box[3], box[4]
        contMask[ymin: ymax, xmin: xmax] = cont_weight
        styMask[ymin: ymax, xmin: xmax] = sty_weight
    
    targetFeature = contMask * cont_feat + styMask * sty_feat
    return targetFeature


def wct_core_random_bbox(cont_feat, sty_feat, bbox, cont_weight=0.7, sty_weight=0.3, device='cpu'):
    _, _, H, W = cont_feat.shape
    bboxXXYY = bbox.detach().type(torch.cuda.IntTensor)

    bboxXXYY[:, 1] = (bbox[:, 1] - bbox[:, 3] / 2) * W
    bboxXXYY[:, 2] = (bbox[:, 2] - bbox[:, 4] / 2) * H
    bboxXXYY[:, 3] = (bbox[:, 1] + bbox[:, 3] / 2) * W
    bboxXXYY[:, 4] = (bbox[:, 2] + bbox[:, 4] / 2) * H

    contMask = torch.ones(H, W)
    contMask = contMask.to(device)

    styMask = torch.zeros(H, W)
    styMask = styMask.to(device)

    for box in bboxXXYY:
        xmin, ymin, xmax, ymax = box[1], box[2], box[3], box[4]

        np.random.seed(0)
        xmin = xmin - np.random.randint(30, 60)
        ymin = ymin - np.random.randint(30, 60)
        xmax = xmax + np.random.randint(30, 60)
        ymax = ymax + np.random.randint(30, 60)

        contMask[ymin: ymax, xmin: xmax] = cont_weight
        styMask[ymin: ymax, xmin: xmax] = sty_weight
    
    targetFeature = contMask * cont_feat + styMask * sty_feat
    return targetFeature


def wct_core_mask(cont_feat, sty_feat, cont_mask, styl_mask, weight=0.3, device='cpu'):
    pass


def wct_core(cont_feat, sty_feat, weight=0.2, registers=None, device='cpu'):
    targetFeature = 0.9 * cont_feat + weight * sty_feat
    return targetFeature


def feature_wct(content_feat, style_feat, content_mask=None, style_mask=None, bbox=None, alpha=1, device='cpu'):
    if config.use_mask:
        target_feature = wct_core_mask(content_feat, style_feat, content_mask, style_mask, device=device)
    elif config.use_bbox:
        target_feature = wct_core_random_bbox(content_feat, style_feat, bbox, device=device)
    else:
        target_feature = wct_core(content_feat, style_feat, device=device)

    target_feature = target_feature.view_as(content_feat)
    target_feature = alpha * target_feature + (1 - alpha) * content_feat # Note that the default alpha is 1.

    return target_feature


class WCT2:
    def __init__(self, 
                 model_path='./model_checkpoints', 
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
    
    def print_(self, msg):
        if self.verbose:
            print(msg)

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

    def transfer(self, content, style, content_mask, style_mask, bbox, alpha=1):
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
                                           content_mask, 
                                           style_mask,
                                           bbox,
                                           alpha=alpha, 
                                           device=self.device)
                # self.print_('transfer at encoder {}'.format(level))

        if 'skip' in self.transfer_at:
            for skip_level in wct2_skip_level:
                for component in [0, 1, 2]:  # component: [LH, HL, HH]
                    content_skips[skip_level][component] = feature_wct(content_skips[skip_level][component], 
                                                                       style_skips[skip_level][component],
                                                                       content_mask, 
                                                                       style_mask,
                                                                       bbox,
                                                                       alpha=alpha, 
                                                                       device=self.device)
                # self.print_('transfer at skip {}'.format(skip_level))

        for level in [4, 3, 2, 1]:
            if 'decoder' in self.transfer_at and level in style_feats['decoder'] and level in wct2_dec_level:
                content_feat = feature_wct(content_feat, 
                                           style_feats['decoder'][level],
                                           content_mask, 
                                           style_mask,
                                           bbox,
                                           alpha=alpha, 
                                           device=self.device)
                # self.print_('transfer at decoder {}'.format(level))
            content_feat = self.decode(content_feat, content_skips, level)

        return content_feat


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def open_image(image_path, image_size=None):
    image = Image.open(image_path)
    if image_size is not None:
        image = transforms.Resize(image_size)(image)
    W, H = image.size

    _transforms = []
    _transforms.append(transforms.CenterCrop((H // 16 * 16, W // 16 * 16)))
    _transforms.append(transforms.ToTensor())
    transform = transforms.Compose(_transforms)

    return transform(image).unsqueeze(0), W, H


def main(config):
    device = 'cpu' if config.cpu or not torch.cuda.is_available() else 'cuda:0'
    device = torch.device(device)

    fnames = os.listdir(config.content)
    for fname in tqdm.tqdm(fnames):
        if not is_image_file(fname):
            print('Invalid file: ', os.path.join(config.content, fname))
            continue

        # Read content image
        content_path = os.path.join(config.content, fname)
        content, c_w, c_h = open_image(content_path, config.image_size)
        content = content.to(device)

        # Read style image
        style_path = config.style
        style, _, _ = open_image(style_path, (c_h, c_w))
        style = style.to(device)

        if not config.transfer_all:
            pass
        else:
            _transfer_at = {'decoder', 'encoder', 'skip'}
            with Timer('Elapsed time in whole WCT: {}', config.verbose):
                output_path = os.path.join(config.output, fname)

                wct2 = WCT2(transfer_at=_transfer_at, 
                            option_unpool=config.option_unpool, 
                            device=device, 
                            verbose=config.verbose)
                
                # transfer the image
                with torch.no_grad():
                    if config.use_mask:
                        pass
                    elif config.use_bbox:
                        label_file = os.path.join(config.label, fname.split('.')[0] + '.txt')

                        bbox = np.loadtxt(label_file, dtype=np.float).reshape(-1, 5)
                        bbox = torch.from_numpy(bbox)
                        bbox = bbox.to(device)

                        img = wct2.transfer(content, 
                                            style, 
                                            content_mask=None, 
                                            style_mask=None,
                                            bbox=bbox,
                                            alpha=config.alpha)
                    else:
                        img = wct2.transfer(content, 
                                            style, 
                                            content_mask=None, 
                                            style_mask=None,
                                            bbox=None,
                                            alpha=config.alpha)
                
                # save the transferred image
                save_image(img.clamp_(0, 1), output_path, padding=0)

                # resize the image to the original size
                img = Image.open(output_path)
                img = img.resize((c_w, c_h))
                img.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=str, default='data/content')
    parser.add_argument('--content_segment', type=str, default=None)
    parser.add_argument('--style', type=str, default='data/style')
    parser.add_argument('--style_segment', type=str, default=None)
    parser.add_argument('--label', type=str, default='data/label')
    parser.add_argument('--output', type=str, default='./outputs')
    parser.add_argument('--image_size', type=int, default=None)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--option_unpool', type=str, default='cat5', choices=['sum', 'cat5'])
    parser.add_argument('--use_mask', type=bool, default=False)
    parser.add_argument('--use_bbox', type=bool, default=False)
    parser.add_argument('-e', '--transfer_at_encoder', action='store_true')
    parser.add_argument('-d', '--transfer_at_decoder', action='store_true')
    parser.add_argument('-s', '--transfer_at_skip', action='store_true')
    parser.add_argument('-a', '--transfer_all', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    config = parser.parse_args()
    print(config)

    if not os.path.exists(os.path.join(config.output)):
        os.makedirs(os.path.join(config.output))
    
    main(config)
