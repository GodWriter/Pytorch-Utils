# Reference: https://github.com/KupynOrest/DeblurGAN

import os
import cv2
import tqdm

import argparse
import numpy as np

from pdb import set_trace as st

parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str, default='data/A')
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str, default='data/B')
parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str, default='data/AB')
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images',type=int, default=100000)
parser.add_argument('--use_AB', dest='use_AB', help='if true: (0001_A, 0001_B) to (0001_AB)',action='store_true')
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg,  getattr(args, arg))

splits = os.listdir(args.fold_A)

for sp in tqdm.tqdm(splits):
    img_fold_A = os.path.join(args.fold_A, sp)
    img_fold_B = os.path.join(args.fold_B, sp)
    img_list = os.listdir(img_fold_A)

    if args.use_AB: 
        img_list = [img_path for img_path in img_list if '_A.' in img_path]

    num_imgs = min(args.num_imgs, len(img_list))
    img_fold_AB = os.path.join(args.fold_AB, sp)

    if not os.path.isdir(img_fold_AB):
        os.makedirs(img_fold_AB)

    for n in tqdm.tqdm(range(num_imgs)):
        name_A = img_list[n]
        path_A = os.path.join(img_fold_A, name_A)

        if args.use_AB:
            name_B = name_A.replace('_A.', '_B.')
        else:
            name_B = name_A

        path_B = os.path.join(img_fold_B, name_B)

        if os.path.isfile(path_A) and os.path.isfile(path_B):
            name_AB = name_A
            path_AB = os.path.join(img_fold_AB, name_AB)

            if args.use_AB:
                name_AB = name_AB.replace('_A.', '.') # remove _A

            im_A = cv2.imread(path_A, cv2.IMREAD_COLOR)
            im_B = cv2.imread(path_B, cv2.IMREAD_COLOR)
            im_AB = np.concatenate([im_A, im_B], 1)

            cv2.imwrite(path_AB, im_AB)
