import os
import tqdm
import imageio

import numpy as np
import imgaug as ia

from imgaug import augmenters as iaa
from PIL import Image


# seq = iaa.Sequential([iaa.CropAndPad(percent=(-0.2, 0.2), pad_mode="edge"),
#                       iaa.ElasticTransformation(alpha=90, sigma=9),
#                       iaa.Cutout()],
#                       random_order=True)

FILE_PATH = "data/lab"
SAVE_PATH = "data/labAug"
SEQ = iaa.Sequential([iaa.ElasticTransformation(alpha=60, sigma=6)], random_order=True)
# SEQ = iaa.Sequential([iaa.EdgeDetect(alpha=0.4, name=None, deterministic=False, random_state=None)], random_order=True)


image_list = os.listdir(FILE_PATH)
for name in tqdm.tqdm(image_list):
    img_path = os.path.join(FILE_PATH, name)
    img = imageio.imread(img_path)

    images_aug = SEQ(images=[img])
    images_aug = Image.fromarray(np.asarray(images_aug[0]))

    images_aug.save(os.path.join(SAVE_PATH, name))

