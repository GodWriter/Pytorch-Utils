import torch
import torch.nn as nn


cfg = {11 : [16,     'M', 32,       'M', 64,  64,            'M', 128, 128,           'M', 128, 128,           'M'],
       13 : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
       16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
       19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}


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
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.vgg = VGG(self.make_layers(cfg[args.vgg_type], batch_norm=True), 
                       num_class=args.num_class).to(args.device)
        
        self.optimizer = torch.optim.Adam(self.vgg.parameters(), lr=args.lr)
        self.loss = nn.CrossEntropyLoss()
    
    def make_layers(self, cfg, batch_norm=False):
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

    def train(self, images, labels):
        self.vgg.train()

        self.optimizer.zero_grad()
        outputs = self.vgg(images)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def valid(self, images, labels):
        self.vgg.eval()

        with torch.no_grad():
            outputs = self.vgg(images)
            _, pred = outputs.topk(3, 1, largest=True, sorted=True)

            labels = labels.view(labels.size(0), -1).expand_as(pred)
            correct = pred.eq(labels).float()

            return correct[:, :3].sum(), correct[:, :1].sum()

        