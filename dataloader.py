import random
import os
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset


class DataLoader(Dataset):
    def __init__(self, list_path, img_size=416):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
                            for path in self.img_files]

        self.img_size = img_size
        self.length = len(self.img_files)

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        # 读取图片
        img_path = self.img_files[index % self.length].rstrip()
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % self.length].rstrip()

        targets = None
        if os.path.exists(label_path):

            # boxes得初始维度为(1, 5)，分别为类别class和bbox坐标
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))

        return img_path, img, targets

    def collate_fn(self, batch):
        pass

    def __len__(self):
        return len(self.img_files)
