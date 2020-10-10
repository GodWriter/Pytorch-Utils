import os
import time
import torch
import argparse

from utils import parse_data_config
from dataloader import DataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=3, help="size of each image batch")
    parser.add_argument("--data_config", type=str, default="config/ships/702.data", help="path to data config file")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")

    opt = parser.parse_args()
    print(opt)

    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    test_path = data_config["test"]

    dataset = DataLoader(train_path)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=True,
                                             num_workers=0,
                                             pin_memory=False)
    
    for batch_i, (img_path, images, labels) in enumerate(dataloader):
        print("batch_i: ", batch_i)
        print("images: ", images.shape)
        print("labels: ", labels.shape)