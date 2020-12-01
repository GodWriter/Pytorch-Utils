import torch
import torch.nn as nn
import torchvision.transforms as transforms

from WCT import WCT2
from PIL import Image


cfg = {11 : [16,     'M', 32,       'M', 64,  64,            'M', 128, 128,           'M', 128, 128,           'M'],
       13 : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
       16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
       19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}


def open_image(img_path):
    transform = transforms.Compose([transforms.Resize((32, 32), Image.BICUBIC),
                                    transforms.ToTensor()])

    return transform(Image.open(img_path).convert('RGB')).unsqueeze(0)


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

        self.action_space = [0.1, 0.2, 0.3, 0.4]
        self.n_actions = len(self.action_space)

        self.style = open_image("data/style/fog1.jpg").to("cuda:0")
        self.wct2 = WCT2(transfer_at={'decoder', 'encoder', 'skip'}, 
                         option_unpool='cat5', 
                         device="cuda:0", 
                         verbose=True)

        self.vgg = VGG(make_layers(cfg[13], batch_norm=True), 
                       num_class=4).to('cuda:0').eval()
        self.vgg.load_state_dict(torch.load("checkpoints/vgg_done.pth"))

    def step(self, state, value):
        # 生成图像
        with torch.no_grad():
            img_aug = self.wct2.transfer(state,
                                         self.style,
                                         alpha=1,
                                         weight=value)
            outputs = self.vgg(img_aug)

            return outputs[:, 2] # 第二维度代表雾天的打分
            
            