import torch
import numpy as np
import torch.nn as nn

from torch.autograd import Variable


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


class Encoder(nn.Module):
    def __init__(self, in_channels=3, dim=64, n_downsample=2):
        super(Encoder, self).__init__()

        layers = [nn.ReflectionPad2d(3),
                  nn.Conv2d(in_channels, dim, 7),
                  nn.InstanceNorm2d(64),
                  nn.LeakyReLU(0.2, inplace=True)]

        # Downsampling
        for _ in range(n_downsample):
            layers += [nn.Conv2d(dim, dim*2, 4, stride=2, padding=1),
                       nn.InstanceNorm2d(dim * 2),
                       nn.ReLU(inplace=True)]
            dim *= 2

        self.model_blocks = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model_blocks(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(in_channels, in_channels, 3),
                                   nn.InstanceNorm2d(in_channels),
                                   nn.ReLU(inplace=True),
                                   nn.ReflectionPad2d(1),
                                   nn.Conv2d(in_channels, in_channels, 3),
                                   nn.InstanceNorm2d(in_channels))

    def forward(self, x):
        return x + self.block(x)


class Decoder(nn.Module):
    def __init__(self, in_channels=3, dim=64, n_upsample=2):
        super(Decoder, self).__init__()

        # Upsampling
        layers = []
        for _ in range(n_upsample):
            layers += [nn.ConvTranspose2d(dim, dim//2, 4, stride=2, padding=1),
                       nn.InstanceNorm2d(dim//2),
                       nn.LeakyReLU(0.2, inplace=True)]
            dim = dim // 2

        # Output layer
        layers += [nn.ReflectionPad2d(3),
                   nn.Conv2d(dim, out_channels, 7),
                   nn.Tanh()]

        self.model_blocks = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model_blocks(x)
        return out


class G_A(nn.Module):
    def __init__(self, out_channels=3, dim=64, shared_E=None, shared_D=None):
        super(G_A, self).__init__()

        layers = []
        self.shared_E = shared_E
        self.shared_D = shared_D

        # residual blocks
        for _ in range(9):
            layers += [ResidualBlock(dim)]

        self.model_blocks = nn.Sequential(*layers)

    def forward(self, x):
        x = self.shared_E(x)
        x = self.model_blocks(x)
        x = self.shared_D(x)

        return x


class G_B(nn.Module):
    def __init__(self, out_channels=3, dim=64, shared_E=None, shared_D=None):
        super(G_B, self).__init__()

        layers = []
        self.shared_E = shared_E
        self.shared_D = shared_D

        # residual blocks
        for _ in range(9):
            layers += [ResidualBlock(dim)]

        self.model_blocks = nn.Sequential(*layers)

    def forward(self, x):
        x = self.shared_E(x)
        x = self.model_blocks(x)
        x = self.shared_D(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        # Calculate output shape of image discriminator (PatchGAN)
        channels, height, width = input_shape
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(*discriminator_block(channels, 64, normalize=False),
                                   *discriminator_block(64, 128),
                                   *discriminator_block(128, 256),
                                   *discriminator_block(256, 512),
                                   nn.Conv2d(512, 1, 3, padding=1))

    def forward(self, img):
        return self.model(img)
