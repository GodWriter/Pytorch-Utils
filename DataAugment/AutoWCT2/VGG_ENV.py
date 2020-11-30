import torch
import torch.nn as nn


cfg = {11 : [16,     'M', 32,       'M', 64,  64,            'M', 128, 128,           'M', 128, 128,           'M'],
       13 : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
       16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
       19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}


def make_layers(cfg, batch_norm=False):
    layers = []
    input_channel = 3

    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]
        if batch_norm:
            layers += [nn.BatchNorm2d(l)]
        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, features, num_class=100):
        super().__init__()

        self.features = features
        self.classifier = nn.Sequential(nn.Linear(512, 1024),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(1024, 1024),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(1024, num_class))

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output


class VGGRapper(object):
    def __init__(self):
        super().__init__()

        self.action_space = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.n_actions = len(self.action_space)

        self.vgg = VGG(make_layers(cfg[13], batch_norm=True), 
                       num_class=4).to('cuda:0').eval()

    def step(self, state, action):
        # 通过action合成图像，并经过图像分类网络给予reward
        pass