import os
import argparse

import torch
import torchvision.transforms as transforms

from PIL import Image

from models.Vgg import VGGRapper
from datasets.WeatherShip import DataLoader


def main(args):
    model = VGGRapper(args)

    transform = transforms.Compose([transforms.Resize((args.img_size, args.img_size), Image.BICUBIC),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = DataLoader(args.train_path, transform)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.n_cpu,
                                               pin_memory=False)

    test_set = DataLoader(args.test_path, transform)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.n_cpu,
                                              pin_memory=False)

    
    for batch_i, (images, labels) in enumerate(train_loader):
        images = images.to(args.device)
        labels = labels.to(args.device)

        loss = model.fit(images, labels)

        n_iter = (args.epochs - 1) * len(train_loader) + batch_i + 1
        writer.add_scalar('Train/loss', loss, n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]'.format(epoch=args.epoch,
                                                                                   trained_samples=batch_i*args.batch_size + len(images),
                                                                                   total_samples=len(train_loader.dataset)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--n_cpu", type=int, default=8, help="dataloader threads number")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--num_class", type=int, default=4, help="number of classes you want to train")
    parser.add_argument("--vgg_type", type=int, default=11, help="you can choose from 11, 13, 16, 19")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--logs", type=str, default="logs/Vgg")
    parser.add_argument("--train_path", type=str, default="data/weather/train.txt", help="txt path saving image paths")
    parser.add_argument("--test_path", type=str, default="data/weather/test.txt", help="txt path saving image paths")
    args = parser.parse_args()
    print(args)

    main(args)