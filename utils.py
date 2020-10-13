import os
import cv2
import tqdm
import random
import shutil

import numpy as np
import xml.etree.ElementTree as ET

from PIL import Image
from xml.dom.minidom import Document
from matplotlib import pyplot as plt


def merge_txt(txt_path, save_path):
    """
    Merge all the txt files to only one txt file.
    """
    txt_list = os.listdir(txt_path)
    total_lines = []

    for txt in tqdm.tqdm(txt_list):
        with open(os.path.join(txt_path, txt), 'r') as fp:
            lines = fp.readlines()
        total_lines.extend(lines)
    
    with open(save_path, 'a+') as fp:
        fp.writelines(total_lines)


def get_file_from_txt(txt_path, file_path, save_path, file_type='img'):
    """
    Get img/label from the txt.

    :param txt_path: path of all the txt files.
    :param file_path: path of all the files that need to be copied.
    :param save_path: path to save the copied files.
    :param file_type: img means copy images, others means copy others, etc. txt mean xxx.txt.
    """

    def get_file(path):
        name = line.split('\n')[0]
        if file_type != 'img':
            name = name.split('.')[0] + '.' + file_type

        old_dir = os.path.join(file_path, name)
        new_dir = os.path.join(save_path, name)

        shutil.copyfile(old_dir, new_dir)
        # os.remove(old_dir)

    txt_list = os.listdir(txt_path)

    for txt in tqdm.tqdm(txt_list):
        with open(os.path.join(txt_path, txt), 'r') as fp:
            lines = fp.readlines()
        for line in lines:
            get_file(line)


def get_percent_txt(percent, file_path, save_path):
    """
    Get a special percent data.
    """

    def get_percent(txt_path):
        with open(txt_path, 'r') as fp:
            lines = fp.readlines()

        # Select random images from the total images.
        nums = len(lines)
        rand_idx = [random.randint(0, nums-1) for _ in range(int(nums * percent))]

        # Get the new lines that should be written to the txt file.
        new_lines = ""
        for idx in rand_idx:
            new_lines += lines[idx]
        
        return new_lines

    name_list = os.listdir(file_path)
    for name in tqdm.tqdm(name_list):
        name_, type_ = name.split('.')

        # txt_path is the original txt, save_txt is the saved percent txt.
        txt_path = os.path.join(file_path, name)
        save_txt = os.path.join(save_path, name_ + '_' + str(int(percent*100)) + '%.' + type_)

        with open(save_txt, 'a+') as fp:
            lines = get_percent(txt_path)
            fp.writelines(lines)


def create_category_txt(category, file_path, save_path):
    """
    Create txt files that contains image names of the same category.
    :param category -> list 
    """
    name_list = os.listdir(file_path)

    # create a dict to save the images files of same category
    cat_dict = {}
    for cat in category:
        cat_dict[cat] = []

    for name in tqdm.tqdm(name_list):
        for cat in category:
            if cat in name:
                cat_dict[cat].append(name)
    
    for cat, names in cat_dict.items():
        with open(os.path.join(save_path, cat + '.txt'), 'a+') as fp:
            lines = ""
            for name in names:
                lines += name + '\n'
            fp.writelines(lines)


def create_dataset_txt(xml_path, txt_path):
    """
    Create the txt files for datasets. 
    We use the xml files rather than images for some images haven't objects.
    """
    names = os.listdir(file_path)

    with open(txt_path, mode='a+') as fp:
        for name in names:
            line = file_path + '/' + name[:-4] + '.jpg' + '\n'
            fp.writelines(line)


def load_classes(path):
    fp = open(path, 'r')
    names = fp.read().split('\n')[:-1]
    return names


def convert_txt_format(txt_path, isLF=True):
    """
    :param isLF: True means converting to Unix(LF), False means converting to Windows(CRLF)
    """

    def to_lf(path, isLF, encoding='utf-8'):
        newline = '\n' if isLF else '\r\n'
        tp = 'Unix(LF)' if isLF else 'Windows(CRLF)'

        with open(path, newline=None, encoding=encoding) as infile:
            str_ = infile.readlines()
            with open(path, 'w', newline=newline, encoding=encoding) as outfile:
                outfile.writelines(str_)
                print("file converting success, format: {0}; encoding: {1}; path: {2}".format(tp, encoding, path))
    
    path_list = os.listdir(txt_path)
    for filename in path_list:
        path = os.path.join(txt_path, filename)
        to_lf(path, isLF)


def xml2txt(xml_path, txt_path):
    """
    Converting xml files to txt files. Num in line means [label, cen_x, cen_y, w, h]
    """
    def getBbox(bndbox, str_list):
        bbox = []
        for text in str_list:
            element = bndbox.find(text)
            bbox.append(int(element.text))
        
        return bbox
    
    xml_file = os.listdir(xml_path)
    for xml in tqdm.tqdm(xml_file):
        # find the xml path and locate the txt save path
        xml_ = os.path.join(xml_path, xml)
        save_path = os.path.join(txt_path, xml.split('.')[0]+'.txt')

        tree = ET.parse(xml_)
        root = tree.getroot()

        # obtain height and width
        size = root.find("size")
        height = float(size.find("height").text)
        width = float(size.find("width").text)

        with open(save_path, mode='a+') as fp:

            # obtain the bbox info
            for object in tree.iter(tag="object"):

                # an object occupy a line, first locate the category info
                line = ""
                name = object.find("name").text
                line += str(class_names.index(name))

                # Then locate the bbox info and changes them to [x, y, w, h]
                bndbox = object.find("bndbox")
                bbox = [float(0) for _ in range(4)]
                xmin, ymin, xmax, ymax = getBbox(bndbox, ["xmin", "ymin", "xmax", "ymax"])

                w = xmax - xmin
                h = ymax - ymin
                center_x = xmin + w // 2
                center_y = ymin + h // 2

                bbox[0] = round(center_x / width, 6)
                bbox[1] = round(center_y / height, 6)
                bbox[2] = round(w / width, 6)
                bbox[3] = round(h / height, 6)

                for box in bbox:
                    line += ' ' + str(box)
                line += '\n'
                fp.writelines(line)


def pltBbox(img_path, label_path):
    img = cv2.imread(img_path)
    boxes = np.loadtxt(label_path, dtype=np.float).reshape(-1, 5)
    boxesXXYY = boxes.copy()
    
    # convert [c_x, c_y, w, h] to [xmin, ymin, xmax, ymax]
    H, W, _ = img.shape
    boxesXXYY[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * W
    boxesXXYY[:, 2] = (boxes[:, 2] - boxes[:, 4] / 2) * H
    boxesXXYY[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * W
    boxesXXYY[:, 4] = (boxes[:, 2] + boxes[:, 4] / 2) * H

    for box in boxesXXYY:
        cv2.rectangle(img, (int(box[1]), int(box[2])), (int(box[3]), int(box[4])), (0, 255, 0), 2)

    cv2.imshow(img_path, img)
    cv2.waitKey()


def parse_data_config(path):
    options = dict()

    options['gpus'] = '0, 1, 2, 3'
    options['num_workers'] = '0'

    with open(path, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue

        key, value = line.split('=')
        options[key.strip()] = value.strip()

    return options


# data_config = parse_data_config("config/ships/702.data")
# class_names = load_classes(data_config["name"])
# xml2txt("data/ships/xmls", "data/ships/labels")

# pltBbox("data/ships/images/1.jpg", "data/ships/labels/1.txt")

# create_dataset_txt("data/ships/images", "config/ships/702-valid.txt")
# convert_txt_format("config/ships/1", isLF=True)

# create_category_txt(['cargoship', 'fishboat', 'daodanhuwei', 'daodanquzhu', 'DSC', 'kechuan', 'passengership', 'qingxinghuwei', 'warship', 'xunyang'], 
#                      "C:/Users/18917/Documents/Python Scripts/pytorch/PyTorch-YOLOv3-master/data/ship/全部船舶数据集/标注版/带增广的四类船舶数据/VOCdevkit/VOC2007/JPEGImages", 
#                      "C:/Users/18917/Documents/Python Scripts/pytorch/PyTorch-YOLOv3-master/data/ship/全部船舶数据集/标注版/带增广的四类船舶数据/VOCdevkit/VOC2007/label-category")


# get_percent_txt(0.1, 
#                 "C:/Users/18917/Documents/Python Scripts/pytorch/PyTorch-YOLOv3-master/data/ship/全部船舶数据集/标注版/带增广的四类船舶数据/VOCdevkit/VOC2007/label-category", 
#                 "C:/Users/18917/Documents/Python Scripts/pytorch/PyTorch-YOLOv3-master/data/ship/全部船舶数据集/标注版/带增广的四类船舶数据/VOCdevkit/VOC2007/label-category-10%")

# get_file_from_txt("C:/Users/18917/Documents/Python Scripts/pytorch/PyTorch-YOLOv3-master/data/ship/全部船舶数据集/标注版/带增广的四类船舶数据/VOCdevkit/VOC2007/label-category-10%",
#                   "C:/Users/18917/Documents/Python Scripts/pytorch/PyTorch-YOLOv3-master/data/ship/全部船舶数据集/标注版/带增广的四类船舶数据/VOCdevkit/VOC2007/labels-four",
#                   "C:/Users/18917/Documents/Python Scripts/pytorch/PyTorch-YOLOv3-master/data/ship/全部船舶数据集/标注版/带增广的四类船舶数据/VOCdevkit/VOC2007/labels-four-10%",
#                   'txt')

# merge_txt("C:/Users/18917/Documents/Python Scripts/pytorch/PyTorch-YOLOv3-master/data/ship/全部船舶数据集/标注版/带增广的四类船舶数据/VOCdevkit/VOC2007/augmentedData/labels-four-10%",
#           "C:/Users/18917/Documents/Python Scripts/pytorch/PyTorch-YOLOv3-master/data/ship/全部船舶数据集/标注版/带增广的四类船舶数据/VOCdevkit/VOC2007/1.txt")