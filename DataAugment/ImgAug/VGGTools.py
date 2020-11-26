import tqdm
import imageio

import numpy as np
import imgaug as ia

from imgaug import augmenters as iaa
from PIL import Image


def augment1(txt_path, save_txt_path):
    """
    Augment data from the txt file.
    """
    seq = iaa.Sequential([iaa.CropAndPad(percent=(-0.2, 0.2), pad_mode="edge"),
                iaa.ElasticTransformation(alpha=90, sigma=9),
                iaa.Cutout()],
                random_order=True)

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