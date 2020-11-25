import os
import argparse

import torch
import torchvision.transforms as transforms

from PIL import Image
from tensorboardX import SummaryWriter

from models.Vgg import VGGRapper
from datasets.WeatherShip import DataLoader


def main(args):
    model = VGGRapper(args)
    writer = SummaryWriter(args.logs)

    transform = transforms.Compose([transforms.Resize((args.img_size, args.img_size), Image.BICUBIC),
                                    transforms.ToTensor()])

    train_set = DataLoader(args.train_path, transform)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.n_cpu,
                                               pin_memory=False)

    valid_set = DataLoader(args.test_path, transform)
    valid_loader = torch.utils.data.DataLoader(valid_set,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.n_cpu,
                                               pin_memory=False)
 
    
    for epoch in range(args.epochs):
        correct_3 = 0.0
        correct_1 = 0.0

        for batch_i, (images, labels) in enumerate(train_loader):
            images = images.to(args.device)
            labels = labels.squeeze(1).to(args.device)

            loss = model.train(images, labels)

            n_iter = epoch * len(train_loader) + batch_i + 1
            writer.add_scalar('Train/loss', loss, n_iter)

            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]'.format(epoch=epoch,
                                                                                       trained_samples=batch_i*args.batch_size + len(images),
                                                                                       total_samples=len(train_loader.dataset)))
        
        if epoch % args.evaluation_interval == 0:
            for batch_i, (images, labels) in enumerate(valid_loader):
                images = images.to(args.device)
                labels = labels.squeeze(1).to(args.device)

                c_3, c_1 = model.valid(images, labels)

                correct_3 += c_3.item()
                correct_1 += c_1.item()
            
            top_1_error = 1 - correct_1 / len(train_loader.dataset)
            top_3_error = 1 - correct_3 / len(train_loader.dataset)

            writer.add_scalar('Valid/Top_1_error', top_1_error, epoch)
            writer.add_scalar('Valid/Top_3_error', top_3_error, epoch)

            print("Top 1 err: ", top_1_error)
            print("Top 3 err: ", top_3_error)

        if epoch % args.checkpoint_interval == 0:
            torch.save(model.vgg.state_dict(), f"checkpoints/vgg_%d.pth" % epoch)
    
    torch.save(model.vgg.state_dict(), f"checkpoints/vgg_done.pth")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--n_cpu", type=int, default=4, help="dataloader threads number")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument('--learning_rate', type=float, default=1e-6)
    parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
    parser.add_argument("--num_class", type=int, default=4, help="number of classes you want to train")
    parser.add_argument("--vgg_type", type=int, default=11, help="you can choose from 11, 13, 16, 19")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--logs", type=str, default="logs/Vgg")
    parser.add_argument("--evaluation_interval", type=int, default=2, help="interval evaluations on validation set")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model weights")
    parser.add_argument("--train_path", type=str, default="data/weather/train.txt", help="txt path saving image paths")
    parser.add_argument("--test_path", type=str, default="data/weather/test.txt", help="txt path saving image paths")
    args = parser.parse_args()
    print(args)

    main(args)