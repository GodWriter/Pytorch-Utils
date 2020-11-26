import os
import tqdm
import numpy as np

import torch

from PIL import Image
from torch.utils.data import Dataset


class DataLoader(Dataset):
    def __init__(self, train_path, transform_=None):
        with open(train_path, "r") as file:
            self.img_files = file.readlines()

        self.length = len(self.img_files)
        self.transform = transform_

    def __getitem__(self, index):
        line = self.img_files[index % self.length].rstrip()
        cat_id, img_path = line.split(' ')

        img = self.transform(Image.open(img_path).convert('RGB')) # 预处理是否需要加入图像正则化？
        label = torch.LongTensor([int(cat_id)])

        return img, label

    def collate_fn(self, batch):
        pass

    def __len__(self):
        return self.length


def create_dataset1(file_path):
    SPLIT = ['train_random.txt', 'test_random.txt']
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


def create_dataset2(file_path):
    SPLIT = ['train.txt', 'test.txt']
    WEATHER = {'0' : 'cloudy', '1' : 'dusky', '2' : 'foggy', '3' : 'sunny'}

    NUM_TRAIN_PER_CAT = 100
    NUM_TEST_PER_CAT = 30

    for cat_id, cat in tqdm.tqdm(WEATHER.items()):
        img_path = file_path + '/' + cat
        img_list = os.listdir(img_path)

        pos = 0 # 用于指示当前文件夹遍历到的图片位置
        size = len(img_list)

        train_path = os.path.join(file_path, SPLIT[0])
        test_path = os.path.join(file_path, SPLIT[1])

        num = 0
        num_train = 0
        num_test = 0

        train_lines = ""
        test_lines = ""

        while num < size and num_train < NUM_TRAIN_PER_CAT and num_test < NUM_TEST_PER_CAT:
            train_line = cat_id + ' ' + img_path + '/' + img_list[num] + '\n'
            test_line = cat_id + ' ' + img_path + '/' + img_list[num+1] + '\n'

            train_lines += train_line
            test_lines += test_line

            num += 2
            num_train += 1
            num_test += 1
        
        while num < size and num_train < NUM_TRAIN_PER_CAT:
            train_line = cat_id + ' ' + img_path + '/' + img_list[num] + '\n'
            train_lines += train_line

            num += 1
            num_train += 1
        
        with open(train_path, 'a+') as fp:
            fp.writelines(train_lines)

        with open(test_path, 'a+') as fp:
            fp.writelines(test_lines)


# create_dataset1('data/weather')