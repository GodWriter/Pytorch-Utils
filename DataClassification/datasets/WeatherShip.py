import os
import tqdm
import random
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset


class DataLoader(Dataset):
    def __init__(self, train_path, img_size=416):
        with open(train_path, "r") as file:
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
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        return img_path, img, targets

    def collate_fn(self, batch):
        pass

    def __len__(self):
        return len(self.img_files)


def create_dataset(file_path):
    SPLIT = ['train.txt', 'test.txt']
    WEATHER = {'0' : 'cloudy', '1' : 'dusky', '2' : 'foggy', '3' : 'sunny'}

    NUM_TRAIN_PER_CAT = 100
    NUM_TEST_PER_CAT = 30

    for cat_id, cat in tqdm.tqdm(WEATHER.items()):
        img_path = file_path + '/' + cat
        img_list = os.listdir(img_path)

        pos = 0 # 用于指示当前文件夹遍历到的图片位置
        size = len(img_list)
        step = size // (NUM_TRAIN_PER_CAT + NUM_TEST_PER_CAT)

        for split, nums in zip(SPLIT, [NUM_TRAIN_PER_CAT, NUM_TEST_PER_CAT]):
            lab_path = os.path.join(file_path, split)

            lines = ""
            while pos < size and nums > 0:
                line = cat_id + ' ' + img_path + '/' + img_list[pos] + '\n'
                lines += line

                pos += step
                nums -= 1
            
            with open(lab_path, 'a+') as fp:
                fp.writelines(lines)


# create_dataset('data/weather')
