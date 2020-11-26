import imageio

import numpy as np
import imgaug as ia

from imgaug import augmenters as iaa
from PIL import Image


image = imageio.imread("data/1.jpg")

seq = iaa.Sequential([iaa.CropAndPad(percent=(-0.2, 0.2), pad_mode="edge"),
                      iaa.ElasticTransformation(alpha=90, sigma=9),
                      iaa.Cutout()],
                      random_order=True)

image_list = [image]

images_aug = seq(images=image_list)
print(images_aug)
ia.imshow(np.hstack(images_aug))

