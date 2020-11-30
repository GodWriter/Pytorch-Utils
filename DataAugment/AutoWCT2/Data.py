import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset


class DataLoader(Dataset):
    def __init__(self, train_path, stride, transform_=None):
        with open(train_path, "r") as file:
            self.img_files = file.readlines()

        self.length = len(self.img_files)
        self.transform = transform_

        self.stride = stride

    def __getitem__(self, index):
        img_path = self.img_files[index % self.length].rstrip()

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img) # 预处理是否需要加入图像正则化？

        c, w, h = img.size()
        n = w // self.stride

        # 将图像从根据网格分割为多个state组成的序列
        img_ = torch.zeros(c, self.stride, h * n)
        for i in range(n):
            img_[:, :, h * i: h * (i+1)] = img[:, self.stride * i: self.stride * (i+1), :]

        return img_

    def collate_fn(self, batch):
        pass

    def __len__(self):
        return self.length