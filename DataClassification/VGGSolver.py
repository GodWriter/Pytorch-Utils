import os
import tqdm
import numpy
import argparse

import torch
import torchvision.transforms as transforms

from PIL import Image
from tensorboardX import SummaryWriter

from models.Vgg import VGGRapper
from datasets.WeatherShip import DataLoader


def train(args):
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
            
            top_1_error = 1 - correct_1 / len(valid_loader.dataset)
            top_3_error = 1 - correct_3 / len(valid_loader.dataset)

            writer.add_scalar('Valid/Top_1_error', top_1_error, epoch)
            writer.add_scalar('Valid/Top_3_error', top_3_error, epoch)

            print("Top 1 err: ", top_1_error)
            print("Top 3 err: ", top_3_error)

        if epoch % args.checkpoint_interval == 0:
            torch.save(model.vgg.state_dict(), f"checkpoints/vgg_%d.pth" % epoch)
    
    torch.save(model.vgg.state_dict(), f"checkpoints/vgg_done.pth")
    writer.close()


def test_chip(args):
    WEATHER = {'0' : '阴天', '1' : '黄昏', '2' : '雾天', '3' : '晴天'}
    
    with open(args.test_path, 'r') as fp:
        lines = fp.readlines()

    model = VGGRapper(args)
    transform = transforms.Compose([transforms.Resize((args.img_size, args.img_size), Image.BICUBIC),
                                    transforms.ToTensor()])
    
    model.vgg.load_state_dict(torch.load(args.weight_path))
    model.vgg.eval()

    correct = 0
    for line in tqdm.tqdm(lines):
        cat_id, img_path = line.rstrip().split(' ')

        img = transform(Image.open(img_path).convert('RGB')).unsqueeze(0).to(args.device)
        outputs = model.vgg(img)

        _, pred = outputs.max(1)
        pred = pred.detach().cpu().numpy()

        if str(pred[0]) == cat_id:
            correct += 1
            print(correct)


def test_list(args):
    IMG_SIZE = 320
    CHIP_NUM = 10
    STRIDE = IMG_SIZE // CHIP_NUM

    with open(args.test_path, 'r') as fp:
        lines = fp.readlines()

    model = VGGRapper(args)
    transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC),
                                    transforms.ToTensor()])
    
    model.vgg.load_state_dict(torch.load(args.weight_path))
    model.vgg.eval()

    correct = 0
    for line in tqdm.tqdm(lines):
        cat_id, img_path = line.rstrip().split(' ')
        img = transform(Image.open(img_path).convert('RGB')).unsqueeze(0).to(args.device)

        labels = [0 for _ in range(4)]
        for row in range(0, IMG_SIZE, STRIDE):
            for col in range(0, IMG_SIZE, STRIDE):
                outputs = model.vgg(img[:, :, row: row+STRIDE, col: col+STRIDE])

                _, pred = outputs.max(1)
                pred = pred.detach().cpu().numpy()[0]

                labels[pred] += 1
        
        if labels.index(max(labels)) == int(cat_id):
            correct += 1
    
    print("Accuracy: ", correct / len(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=str, default="test_list", help="you can choose from train, test_list, test_chip, test_single")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--n_cpu", type=int, default=8, help="dataloader threads number")
    parser.add_argument("--batch_size", type=int, default=64, help="size of each image batch")
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--num_class", type=int, default=4, help="number of classes you want to train")
    parser.add_argument("--vgg_type", type=int, default=13, help="you can choose from 11, 13, 16, 19")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--logs", type=str, default="logs/Vgg/vgg-chips-bs64")
    parser.add_argument("--evaluation_interval", type=int, default=2, help="interval evaluations on validation set")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model weights")
    parser.add_argument("--train_path", type=str, default="data/weather/chips_train.txt", help="txt path saving image paths")
    parser.add_argument("--test_path", type=str, default="data/weather/test.txt", help="txt path saving image paths")
    parser.add_argument("--test_image", type=str, default="data/weather/test7.jpg", help="test image path")
    parser.add_argument("--weight_path", type=str, default="checkpoints/vgg-chips-adam-bs64/vgg_done.pth", help="model to test")
    args = parser.parse_args()
    print(args)

    if args.module == "train":
        train(args)
    elif args.module == "test_list":
        test_list(args)
    elif args.module == "test_chip":
        test_chip(args)