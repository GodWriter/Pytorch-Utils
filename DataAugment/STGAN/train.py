import os
import time
import torch
import itertools
import datetime

from torch.autograd import Variable

from config import parse_args
from utils import save_sample
from model import Encoder, Decoder, GeneratorA, GeneratorB, Discriminator, weights_init_normal
from dataloader import coco_loader


def train():
    opt = parse_args()
    cuda = True if torch.cuda.is_available() else False

    input_shape = (opt.channels, opt.img_width, opt.img_height)
    FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # get dataloader
    train_loader = coco_loader(opt, mode='train')
    test_loader = coco_loader(opt, mode='test')

    # Initialize two generators and the discriminator
    shared_E = Encoder(opt.channels, opt.dim, opt.n_downsample)
    shared_D = Decoder(3, 256, opt.n_upsample)

    G_A = GeneratorA(opt.n_residual, 256, shared_E, shared_D)
    G_B = GeneratorB(opt.n_residual, 256, shared_E, shared_D)

    D_B = Discriminator(input_shape)

    # Initialize weights
    shared_E.apply(weights_init_normal)
    shared_D.apply(weights_init_normal)
    G_A.apply(weights_init_normal)
    G_B.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

    if cuda:
        shared_E = shared_E.cuda()
        shared_D = shared_D.cuda()
        G_A = G_A.cuda()
        G_B = G_B.cuda()
        D_B = D_B.cuda()

    for batch_i, img in enumerate(train_loader):
        img = Variable(img.type(FloatTensor))

        stylized_img = G_A(img)
        reconstructed_img = G_B(stylized_img)



    # # Loss function
    # adversarial_loss = torch.nn.MSELoss()
    # pixel_loss = torch.nn.L1Loss()

    # # Optimizers
    # optimizer_G = torch.optim.Adam(itertools.chain(E1.parameters(), E2.parameters(), G1.parameters(), G2.parameters()),
    #                                lr=opt.lr,
    #                                betas=(opt.b1, opt.b2))
    # optimizer_D1 = torch.optim.Adam(D1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    # optimizer_D2 = torch.optim.Adam(D2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # # Learning rate update schedulers
    # lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.epochs, 0, opt.decay_epoch).step)
    # lr_scheduler_D1 = torch.optim.lr_scheduler.LambdaLR(optimizer_D1, lr_lambda=LambdaLR(opt.epochs, 0, opt.decay_epoch).step)
    # lr_scheduler_D2 = torch.optim.lr_scheduler.LambdaLR(optimizer_D2, lr_lambda=LambdaLR(opt.epochs, 0, opt.decay_epoch).step)

    # prev_time = time.time()
    # for epoch in range(opt.epochs):
    #     for i, (img_A, img_B) in enumerate(train_loader):

    #         # Model inputs
    #         X1 = Variable(img_A.type(FloatTensor))
    #         X2 = Variable(img_B.type(FloatTensor))

    #         # Adversarial ground truths
    #         valid = Variable(FloatTensor(img_A.shape[0], *D1.output_shape).fill_(1.0), requires_grad=False)
    #         fake = Variable(FloatTensor(img_A.shape[0], *D1.output_shape).fill_(0.0), requires_grad=False)

    #         # -----------------------------
    #         # Train Encoders and Generators
    #         # -----------------------------

    #         # Get shared latent representation
    #         mu1, Z1 = E1(X1)
    #         mu2, Z2 = E2(X2)

    #         # Reconstruct images
    #         recon_X1 = G1(Z1)
    #         recon_X2 = G2(Z2)

    #         # Translate images
    #         fake_X1 = G1(Z2)
    #         fake_X2 = G2(Z1)

    #         # Cycle translation
    #         mu1_, Z1_ = E1(fake_X1)
    #         mu2_, Z2_ = E2(fake_X2)
    #         cycle_X1 = G1(Z2_)
    #         cycle_X2 = G2(Z1_)

    #         # Losses for encoder and generator
    #         id_loss_1 = opt.lambda_id * pixel_loss(recon_X1, X1)
    #         id_loss_2 = opt.lambda_id * pixel_loss(recon_X2, X2)

    #         adv_loss_1 = opt.lambda_adv * adversarial_loss(D1(fake_X1), valid)
    #         adv_loss_2 = opt.lambda_adv * adversarial_loss(D2(fake_X2), valid)

    #         cyc_loss_1 = opt.lambda_cyc * pixel_loss(cycle_X1, X1)
    #         cyc_loss_2 = opt.lambda_cyc * pixel_loss(cycle_X2, X2)

    #         KL_loss_1 = opt.lambda_KL1 * compute_KL(mu1)
    #         KL_loss_2 = opt.lambda_KL1 * compute_KL(mu2)
    #         KL_loss_1_ = opt.lambda_KL2 * compute_KL(mu1_)
    #         KL_loss_2_ = opt.lambda_KL2 * compute_KL(mu2_)

    #         # total loss for encoder and generator
    #         G_loss = id_loss_1 + id_loss_2 \
    #                  + adv_loss_1 + adv_loss_2 \
    #                  + cyc_loss_1 + cyc_loss_2 + \
    #                  KL_loss_1 + KL_loss_2 + KL_loss_1_ + KL_loss_2_

    #         G_loss.backward()
    #         optimizer_G.step()

    #         # ----------------------
    #         # Train Discriminator 1
    #         # ----------------------

    #         optimizer_D1.zero_grad()

    #         D1_loss = adversarial_loss(D1(X1), valid) + adversarial_loss(D1(fake_X1.detach()), fake)
    #         D1_loss.backward()

    #         optimizer_D1.step()

    #         # ----------------------
    #         # Train Discriminator 2
    #         # ----------------------

    #         optimizer_D2.zero_grad()

    #         D2_loss = adversarial_loss(D2(X2), valid) + adversarial_loss(D2(fake_X2.detach()), fake)
    #         D2_loss.backward()

    #         optimizer_D2.step()

    #         # ------------------
    #         # Log Information
    #         # ------------------

    #         batches_done = epoch * len(train_loader) + i
    #         batches_left = opt.epochs * len(train_loader) - batches_done
    #         time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
    #         prev_time = time.time()

    #         print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] ETA: %s" %
    #               (epoch, opt.epochs, i, len(train_loader), (D1_loss + D2_loss).item(), G_loss.item(), time_left))

    #         if batches_done % opt.sample_interval == 0:
    #             save_sample(opt.dataset, test_loader, batches_done, E1, E2, G1, G2, FloatTensor)

    #         if batches_done % opt.checkpoint_interval == 0:
    #             torch.save(E1.state_dict(), "checkpoints/%s/E1_%d.pth" % (opt.dataset, epoch))
    #             torch.save(E2.state_dict(), "checkpoints/%s/E2_%d.pth" % (opt.dataset, epoch))
    #             torch.save(G1.state_dict(), "checkpoints/%s/G1_%d.pth" % (opt.dataset, epoch))
    #             torch.save(G2.state_dict(), "checkpoints/%s/G2_%d.pth" % (opt.dataset, epoch))

    #     # Update learning rates
    #     lr_scheduler_G.step()
    #     lr_scheduler_D1.step()
    #     lr_scheduler_D2.step()

    # torch.save(shared_E.state_dict(), "checkpoints/%s/shared_E_done.pth" % opt.dataset)
    # torch.save(shared_G.state_dict(), "checkpoints/%s/shared_G_done.pth" % opt.dataset)
    # torch.save(E1.state_dict(), "checkpoints/%s/E1_done.pth" % opt.dataset)
    # torch.save(E2.state_dict(), "checkpoints/%s/E2_done.pth" % opt.dataset)
    # torch.save(G1.state_dict(), "checkpoints/%s/G1_done.pth" % opt.dataset)
    # torch.save(G2.state_dict(), "checkpoints/%s/G2_done.pth" % opt.dataset)
    # print("Training Process has been Done!")


if __name__ == '__main__':
    train()
