import os
import glob
import random

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None):
        self.transform = transforms_
        self.files = glob.glob(r"%s/*.*"%root)

        self.length = len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index % self.length])

        if img.mode != "RGB": img = img.convert("RGB")
        img = self.transform(img)

        return img

    def __len__(self):
        return self.length


def coco_loader(opt, mode, transform):
    loader = ImageDataset(opt.dataset,
                          transforms_=transform)

    # create the data_loader
    if mode == 'train':
        data_loader = DataLoader(loader,
                                 batch_size=opt.batch_size,
                                 shuffle=True)
    elif mode == 'test':
        data_loader = DataLoader(loader,
                                 batch_size=4,
                                 shuffle=True)

    return data_loader
