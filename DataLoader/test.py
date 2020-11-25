import os
import time
import torch
import argparse

from dataloader import DataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=3, help="size of each image batch")
    parser.add_argument("--train_path", type=str, default="config/train.txt", help="txt path saving image paths")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    dataset = DataLoader(opt.train_path)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=True,
                                             num_workers=0,
                                             pin_memory=False)
    
    for batch_i, (img_path, images, labels) in enumerate(dataloader):
        print("batch_i: ", batch_i)
        print("images: ", images)
        print("labels: ", labels)

        print("=" * 50)