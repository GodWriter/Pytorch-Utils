import os
import tqdm
import imageio

import numpy as np
import imgaug as ia

from imgaug import augmenters as iaa
from PIL import Image


def create_gif(file_path):
    gif_name = os.path.join(file_path, 'gif.gif')

    frames = []
    for i in tqdm.tqdm(range(110)):
        img = Image.open(os.path.join(file_path, str(i) + '.jpg'))
        frames.append(img)

    imageio.mimsave(gif_name, frames, 'GIF', duration=0.1)


def get_chips_augment(txt_path, img_size=640, chip_num=10, save_name="chips", save_txt_path="chips.txt"):
    """
    Get chips of the images.
    """
    STRIDE = img_size // chip_num

    with open(txt_path, 'r') as fp:
        lines = fp.readlines()
    
    new_lines = []
    for line in tqdm.tqdm(lines):
        cat_id, img_path = line.rstrip().split(' ')

        image = np.asarray(Image.open(img_path).convert('RGB').resize((img_size, img_size)))
        img_path = img_path.replace('cloudy', save_name).replace('dusky', save_name).replace('foggy', save_name).replace('sunny', save_name)

        for row in range(0, img_size, STRIDE):
            for col in range(0, img_size, STRIDE):
                new_path = img_path[:-4] + '_' + str(row) + '_' + str(col) + img_path[-4:]
                new_line = cat_id + ' ' + new_path + '\n'

                new_img = image[row: row+STRIDE, col: col+STRIDE, :]
                new_img = Image.fromarray(new_img)
                new_img.save(new_path)

                new_lines.append(new_line)

    with open(save_txt_path, 'a+') as fp:
        fp.writelines(new_lines)


def augment1(txt_path, save_txt_path):
    """
    Augment data from the txt file.
    """
    seq = iaa.Sequential([iaa.ElasticTransformation(alpha=90, sigma=9),
                          iaa.Cutout()],random_order=False)

    with open(txt_path, 'r') as fp:
        lines = fp.readlines()
    
    new_lines = []
    for line in tqdm.tqdm(lines):
        new_line = ""

        cat_id, img_path = line.rstrip().split(' ')
        new_line += cat_id

        image = imageio.imread(img_path)
        images_aug = seq(images=[image])[0]

        img_path = img_path.replace('cloudy', 'augment').replace('dusky', 'augment').replace('foggy', 'augment').replace('sunny', 'augment')
        new_line += ' ' + img_path + '\n'
        new_lines.append(new_line)

        images_aug = Image.fromarray(images_aug)
        images_aug.save(img_path)

    with open(save_txt_path, 'a+') as fp:
        fp.writelines(new_lines)


# augment1("data/weather/train.txt", "data/weather/augment.txt")
# get_chips_augment(txt_path="data/weather/test.txt", save_name="chips_test", save_txt_path="data/weather/chips_test.txt")
# create_gif("C:/Users/18917/Documents/Python Scripts/pytorch/Lab/Pix2Pix-forlab/data-blur/6")