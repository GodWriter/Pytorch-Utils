import os
import glob
import random
import imageio
import numpy as np

import torch
import torch.nn as nn

from PIL import Image
from torchvision import models
from torch.autograd import Variable
from torchvision.utils import save_image, make_grid


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.select = ['0', '5', '10', '19', '28']
        self.vgg = models.vgg19(pretrained=True).features
    
    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features

    def calc_mean_std(self, feat, eps=1e-5):
        N, C = feat.size()[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std
  
    def compute_st_loss(self, target, content, style, style_weight):
        style_loss = 0.0
        content_loss = 0.0

        for f1, f2, f3 in zip(target, content, style):
            content_loss += torch.mean((f1 - f2)**2)

            target_mean, target_std = self.calc_mean_std(f1)
            style_mean, style_std = self.calc_mean_std(f3)

            mean_loss = torch.mean((target_mean - style_mean)**2)
            std_loss = torch.mean((target_std - style_std)**2)
            style_loss += mean_loss + std_loss

        st_loss = content_loss + style_weight * style_loss

        return st_loss


def load_img(img_path, transform=None):
    img = Image.open(img_path)
    if transform:
        img = transform(img).unsqueeze(0)
    return img


def stack_img(image_path):
    imgs = []

    files = sorted(glob.glob("%s/*.*" % image_path))
    for file in files:
        imgs.append(np.array(Image.open(file)))

    result_img = np.vstack(tuple(imgs))
    Image.fromarray(result_img).save(os.path.join(image_path, 'result.png'))


def create_gif(image_path):
    frames = []
    gif_name = os.path.join("images", 'display2.gif')
    image_list = os.listdir(image_path)

    image_id = []
    for name in image_list:
        image_id.append(name[:-4])
    sorted(image_id)

    cnt = 0
    for idx in image_id:
        if cnt % 5 == 0:
            frames.append(imageio.imread(os.path.join(image_path, str(idx) + '.png')))
        cnt += 1
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.1)


def resize_img(path):
    names = os.listdir(path)
    for name in names:
        img_path = os.path.join(path, name)
        img = Image.open(img_path)
        img = img.resize((172, 172))
        img.save(img_path)


def save_sample(style_name, test_loader, batches_done, G_A, G_B, FloatTensor):
    img = next(iter(test_loader))
    img = Variable(img.type(FloatTensor))

    stylized_img = G_A(img)
    reconstructed_img = G_B(stylized_img)

    samples = torch.cat((stylized_img.data, reconstructed_img.data), 0)
    save_image(samples, "images/%s/%d.png" % (style_name, batches_done), nrow=4, normalize=True)


if __name__ == "__main__":
    # image_path = "images/example2"
    # # resize_img(image_path)
    # create_gif(image_path)

    # vgg = VGGNet().to('cuda:0').eval()

    pass