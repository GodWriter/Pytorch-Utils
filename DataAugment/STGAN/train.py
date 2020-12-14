import os
import time
import torch
import itertools
import datetime

import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torch.autograd import Variable

from config import parse_args
from utils import VGGNet, save_sample, load_img
from dataloader import coco_loader
from model import Encoder, Decoder, GeneratorA, GeneratorB, Discriminator, weights_init_normal, LambdaLR


def train():
    opt = parse_args()
    cuda = True if torch.cuda.is_available() else False

    input_shape = (opt.channels, opt.img_width, opt.img_height)
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    transform = transforms.Compose([transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
                                    transforms.RandomCrop((opt.img_height, opt.img_width)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Get dataloader
    train_loader = coco_loader(opt, mode='train', transform=transform)
    test_loader = coco_loader(opt, mode='test', transform=transform)

    # Get vgg
    vgg = VGGNet()

    # Initialize two generators and the discriminator
    shared_E = Encoder(opt.channels, opt.dim, opt.n_downsample)
    shared_D = Decoder(3, 256, opt.n_upsample)

    G_A = GeneratorA(opt.n_residual, 256, shared_E, shared_D)
    G_B = GeneratorB(opt.n_residual, 256, shared_E, shared_D)

    D_B = Discriminator(input_shape)

    # Initialize weights
    G_A.apply(weights_init_normal)
    G_B.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixel = torch.nn.L1Loss()

    if cuda:
        vgg = vgg.cuda().eval()
        G_A = G_A.cuda()
        G_B = G_B.cuda()
        D_B = D_B.cuda()
        criterion_GAN.cuda()
        criterion_pixel.cuda()
    
    optimizer_G = torch.optim.Adam(itertools.chain(G_A.parameters(), G_B.parameters()), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    # Compute the style features in advance
    style_img = Variable(load_img(opt.style_img, transform).type(FloatTensor))
    style_feature = vgg(style_img)

    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):
        for batch_i, content_img in enumerate(train_loader):
            content_img = Variable(content_img.type(FloatTensor))

            valid = Variable(FloatTensor(np.ones((content_img.size(0), *D_B.output_shape))), requires_grad=False)
            fake = Variable(FloatTensor(np.zeros((content_img.size(0), *D_B.output_shape))), requires_grad=False)

            # ---------------------
            #  Train Generators
            # ---------------------

            optimizer_G.zero_grad()

            # 生成的图像并没有做反正则化，得保证：内容，风格，生成图，图像预处理的一致性！
            stylized_img = G_A(content_img)

            target_feature = vgg(stylized_img)
            content_feature = vgg(content_img)
            loss_st = opt.lambda_st * vgg.compute_st_loss(target_feature, content_feature, style_feature, opt.lambda_style)

            reconstructed_img = G_B(stylized_img)
            loss_adv = opt.lambda_adv * criterion_GAN(D_B(reconstructed_img), valid)

            loss_G = loss_st + loss_adv
            loss_G.backward()
            optimizer_G.step()

            # ----------------------
            #  Train Discriminator
            # ----------------------

            optimizer_D.zero_grad()

            loss_D = criterion_GAN(D_B(content_img), valid) + criterion_GAN(D_B(reconstructed_img.detach()), fake)
            loss_D.backward()
            optimizer_D.step()

            # ------------------
            # Log Information
            # ------------------

            batches_done = epoch * len(train_loader) + batch_i
            batches_left = opt.n_epochs * len(train_loader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] ETA: %s" %
                    (epoch, opt.n_epochs, batch_i, len(train_loader), loss_D.item(), loss_G.item(), time_left))

            if batches_done % opt.sample_interval == 0:
                save_sample(opt.style_name, test_loader, batches_done, G_A, G_B, FloatTensor)

            if batches_done % opt.checkpoint_interval == 0:
                torch.save(G_A.state_dict(), "checkpoints/%s/G_A_%d.pth" % (opt.style_name, epoch))
                torch.save(G_B.state_dict(), "checkpoints/%s/G_B_%d.pth" % (opt.style_name, epoch))

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D.step()

    torch.save(G_A.state_dict(), "checkpoints/%s/G_A_done.pth" % opt.style_name)
    torch.save(G_B.state_dict(), "checkpoints/%s/G_B_done.pth" % opt.style_name)
    print("Training Process has been Done!")


if __name__ == '__main__':
    train()
