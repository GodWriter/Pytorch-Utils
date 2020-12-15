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


def convert_to_single_class_txt(txt_path, save_path):
    """
    Convert the multiple classes labels to single label.
    """
    names = os.listdir(txt_path)
    dicts = {}

    for name in tqdm.tqdm(names):
        file_path = os.path.join(txt_path, name)

        with open(file_path, 'r') as fp:
            lines = fp.readlines()
        
        for idx in range(len(lines)):
            lines[idx] = '0' + lines[idx][1:]
        
        with open(os.path.join(save_path, name), 'a+') as fp:
            fp.writelines(lines)


def print_txt_category(txt_path):
    """
    Find the wrong category from all the txts.
    """
    names = os.listdir(txt_path)
    dicts = {}

    for name in tqdm.tqdm(names):
        file_path = os.path.join(txt_path, name)
        bbox = np.loadtxt(file_path, dtype=np.float).reshape(-1, 5)

        for box in bbox:
            if box[0] not in dicts:
                dicts[box[0]] = 1
            else:
                dicts[box[0]] += 1
    
    print(dicts)


def getAddedData(txt_path, save_path, prefix):
    """
    Find the source name of augmented data.
    """
    with open(txt_path, 'r') as fp:
        lines = fp.readlines()
    
    newLines = []
    for line in tqdm.tqdm(lines):
        name = line.split('/')[-1].split('-')[1:]

        line = ""
        for n in name[:-1]:
            line += n + '-'
        line += name[-1]
        line = prefix + line

        newLines.append(line)
    
    with open(save_path, 'a+') as fp:
        fp.writelines(newLines)
        

def createBash(command, times, save_path):
    """
    Create bash commands to solve the stopped problem.
    """
    lines = ""

    for idx in tqdm.tqdm(range(0, times+1, 10)):
        lines += command.replace('*', str(idx)) + '\n'
    
    with open(save_path, 'a+') as fp:
        fp.writelines(lines)


def txt2xml(class_names, img_path, txt_path, xml_path):
    """
    Converting txt files to xml files.
    """
    def createNode(name, value):
        node = doc.createElement(name)
        if value is not None:
            node_text = doc.createTextNode(value)
            node.appendChild(node_text)
        return node

    img_list = os.listdir(img_path)
    for img_name in tqdm.tqdm(img_list):
        img = cv2.imread(os.path.join(img_path, img_name))
        H, W, _ = img.shape

        boxes = np.loadtxt(os.path.join(txt_path, img_name.split('.')[0] + '.txt'), dtype=np.float).reshape(-1, 5)
        boxesXXYY = boxes.copy()

        boxesXXYY[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * W
        boxesXXYY[:, 2] = (boxes[:, 2] - boxes[:, 4] / 2) * H
        boxesXXYY[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * W
        boxesXXYY[:, 4] = (boxes[:, 2] + boxes[:, 4] / 2) * H

        # create xmls
        doc = Document()
        annotation = doc.createElement("annotation")

        annotation.appendChild(createNode("folder", "702data"))
        annotation.appendChild(createNode("filename", img_name))
        annotation.appendChild(createNode("source", "None"))

        # create node of size
        size = createNode("size", None)
        size.appendChild(createNode("width", str(W)))
        size.appendChild(createNode("height", str(H)))
        size.appendChild(createNode("depth", str(3)))
        annotation.appendChild(size)

        for box in boxesXXYY:
            object_ = createNode("object", None)
            object_.appendChild(createNode("name", class_names[int(box[0])]))
            object_.appendChild(createNode("pose", "Unspecified"))
            object_.appendChild(createNode("truncated", str(0)))
            object_.appendChild(createNode("difficult", str(0)))
            
            # create bounding box
            bndbox = createNode("bndbox", None)
            bndbox.appendChild(createNode("xmin", str(int(box[1]))))
            bndbox.appendChild(createNode("ymin", str(int(box[2]))))
            bndbox.appendChild(createNode("xmax", str(int(box[3]))))
            bndbox.appendChild(createNode("ymax", str(int(box[4]))))
            object_.appendChild(bndbox)
            
            annotation.appendChild(object_)

        doc.appendChild(annotation)

        with open(os.path.join(xml_path, img_name.split('.')[0] + '.xml'), 'wb') as f:
            f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))


def get_random_dataset(img_path, save_path, ratio):
    """
    Making training set and testing set.
    """
    img_list = os.listdir(img_path)
    img_len = len(img_list)

    train_set = []
    test_set = []

    for i in tqdm.tqdm(range(img_len)):
        img_dir = img_path + '/' + img_list[i] + '\n'

        if i % ratio == 0:
            test_set.append(img_dir)
        else:
            train_set.append(img_dir)
    
    with open(os.path.join(save_path, "train.txt"), 'a+') as fp:
        fp.writelines(train_set)
    
    with open(os.path.join(save_path, "test.txt"), 'a+') as fp:
        fp.writelines(test_set)


# def get_random_dataset(img_path, save_path, ratio):
#     """
#     Making training set and testing set.
#     @param ratio: precent of the training set.
#     """
#     img_list = os.listdir(img_path)
#     img_len = len(img_list)

#     train_size = int(img_len * ratio) # size of the training set.
#     random_list = [i for i in range(img_len)] # which is used to choose random images.
#     record_list = [0 for _ in range(img_len)] # which is used to record whether the image is used for training or testing. 0 means test

#     while train_size:
#         idx = random.randint(0, img_len-1)
#         record_list[idx] = 1

#         # replace [idx] with [img_len - 1]
#         tmp = random_list[idx]
#         random_list[idx] = random_list[img_len-1]
#         random_list[img_len-1] = tmp 

#         train_size -= 1
#         img_len -= 1
    
#     print("record_list: ", record_list)


def resize_images(img_path, save_path, new_size=416):
    """
    Resize images and save to the new path.
    """
    img_list = os.listdir(img_path)

    for img_name in tqdm.tqdm(img_list):
        old_dir = os.path.join(img_path, img_name)
        new_dir = os.path.join(save_path, img_name)

        img = Image.open(old_dir)

        # resize the image
        img = img.resize((new_size, new_size))
        img.save(new_dir)


def delete_null_files(txt_path, img_path):
    """
    Check whether the txt file is null, and delete the corresponsding images and txts.
    """
    names = os.listdir(txt_path)

    for name in tqdm.tqdm(names):
        file_path = os.path.join(txt_path, name)

        if not os.path.getsize(file_path):
            # delete the txt
            os.remove(file_path)

            # delete the image
            os.remove(os.path.join(img_path, name.split('.')[0] + '.jpg'))


def rename_xml(xml_path, prefix):
    """
    Rename the xml and filename in it.
    """

    # First rename the xml.
    xml_names = os.listdir(xml_path)

    for name in tqdm.tqdm(xml_names):
        old_name = os.path.join(xml_path, name)
        new_name = os.path.join(xml_path, prefix + name)

        os.rename(old_name, new_name)
    
    # Then rename the filename
    xml_names = os.listdir(xml_path)

    for name in tqdm.tqdm(xml_names):
        # Rename the xml file.
        xml_dir = os.path.join(xml_path, name)

        tree = ET.parse(xml_dir)
        root = tree.getroot()
        
        element_filename = root.find("filename")
        element_filename.text = prefix + element_filename.text

        tree.write(xml_dir)
    

def check(xml_path, img_path):
    """
    Check whether img and label is paired.
    """
    img_list = os.listdir(img_path)

    for img_name in tqdm.tqdm(img_list):
        name, format = img_name.split('.')

        img_dir = os.path.join(img_path, img_name)
        xml_dir = os.path.join(xml_path, name + '.txt')

        if os.path.exists(img_dir) and os.path.exists(xml_dir):
            continue
        else:
            print(img_dir)
            break


def process_data_size(xml_path, img_path):
    """
    To resize the image and resize the xml.
    """
    def get_element(ele, str_list):
        element_list = []
        for text in str_list:
            element = ele.find(text)
            element_list.append(element)
        
        return element_list

    def do_operation(element_list, ratio):
        for element in element_list:
            value = int(element.text) // ratio
            element.text = str(value)

    img_list = os.listdir(img_path)
    for img_name in tqdm.tqdm(img_list):
        name, format = img_name.split('.')

        img_dir = os.path.join(img_path, img_name)
        xml_dir = os.path.join(xml_path, name + '.xml')
        
        img = Image.open(img_dir)
        W, H = img.size

        if W > 800 or H > 800:
            ratio = max(W // 800, 1) # avoiding the case of 0
            shape = (W // ratio, H // ratio)

            # resize the image
            img = img.resize(shape)
            img.save(img_dir)

            # modify the xml
            tree = ET.parse(xml_dir)
            root = tree.getroot()

            # modify the width and height
            for elem in tree.iter(tag='height'):
                value = int(elem.text) // ratio
                elem.text = str(value)

            for elem in tree.iter(tag='width'):
                value = int(elem.text) // ratio
                elem.text = str(value)

            # modify the bbox
            for elem in tree.iter(tag='bndbox'):
                element_list = get_element(elem, ["xmin", "xmax", "ymin", "ymax"])
                do_operation(element_list, ratio)
            
            tree.write(xml_dir)


def process_data_name(xml_path, img_path):
    """
    To get the cleanest images and labels.
    """
    img_list = os.listdir(img_path)

    for img_name in tqdm.tqdm(img_list):
        name, format = img_name.split('.')

        if format != 'jpg':
            new_name = name + '.jpg'

            # Rename the image file.
            old_dir = os.path.join(img_path, img_name)
            new_dir = os.path.join(img_path, new_name)

            img = Image.open(old_dir)
            img = img.convert('RGB')
            img.save(old_dir)

            # It's a surprising thing in windows that .JPG is the .jpg
            os.rename(old_dir, new_dir)

            # Rename the xml file.
            xml_dir = os.path.join(xml_path, name + '.xml')

            tree = ET.parse(xml_dir)
            root = tree.getroot()
            
            element_filename = root.find("filename")
            element_filename.text = new_name

            tree.write(xml_dir)


def remove_label_invalid(xml_path, img_path, invalid_label_path):
    """
    According to total img files, removing invalid labels.
    """
    xml_list = os.listdir(xml_path)
    img_list = os.listdir(img_path)

    xml_name = [xml.split('.')[0] for xml in xml_list]
    img_name = [img.split('.')[0] for img in img_list]

    for idx in tqdm.tqdm(range(len(xml_name))):
        if xml_name[idx] not in img_name:
            old_dir = os.path.join(xml_path, xml_list[idx])
            new_dir = os.path.join(invalid_label_path, xml_list[idx])

            shutil.copy(old_dir, new_dir)
            os.remove(old_dir)


def remove_image_unlabelled(xml_path, img_path, unlabelled_img_path):
    """
    According to total xml files, removing images unlabelled.
    """
    xml_list = os.listdir(xml_path)
    img_list = os.listdir(img_path)

    xml_name = [xml.split('.')[0] for xml in xml_list]
    img_name = [img.split('.')[0] for img in img_list]

    for idx in tqdm.tqdm(range(len(img_name))):
        if img_name[idx] not in xml_name:
            old_dir = os.path.join(img_path, img_list[idx])
            new_dir = os.path.join(unlabelled_img_path, img_list[idx])

            shutil.copy(old_dir, new_dir)
            os.remove(old_dir)


def modify_txt_file(txt_path, save_path, prefix):
    """
    Method to modify the txt lines. etc. prefix + name
    """
    with open(txt_path, 'r') as fp:
        lines = fp.readlines()
    
    for idx in range(len(lines)):
        lines[idx] = prefix + lines[idx]
    
    with open(save_path, 'a+') as fp:
        fp.writelines(lines)


def rename_file(file_path, prefix):
    """
    Method to rename the file. etc. prefix + name
    """
    file_names = os.listdir(file_path)

    for name in tqdm.tqdm(file_names):
        old_name = os.path.join(file_path, name)
        new_name = os.path.join(file_path, prefix + name)

        os.rename(old_name, new_name)


def modify_img_size(txt_path, file_path, shape=(416, 416)):
    """
    Resize the image and modify its label at the same time.
    """

    with open(txt_path, 'r') as fp:
        lines = fp.readlines()
    
    for line in tqdm.tqdm(lines):
        name = line.split('\n')[0]
        img_path = os.path.join(file_path, name)

        # read img and get its shape
        img = Image.open(img_path)
        W, H = img.size

        if W > 800 or H > 800:
            # resize the image
            img = img.resize(shape)
            img.save(img_path)


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


def create_dataset_txt(file_path, txt_path):
    """
    Create the txt files for datasets. 
    We use the xml files rather than images for some images haven't objects.
    """
    names = os.listdir(file_path)
    prefix = file_path.replace('\\', '/')

    lines = []
    for name in tqdm.tqdm(names):
        line = prefix + '/' + name + '\n'
        lines.append(line)

    with open(txt_path, mode='a+') as fp:
        fp.writelines(lines)


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
                # print("file converting success, format: {0}; encoding: {1}; path: {2}".format(tp, encoding, path))
    
    path_list = os.listdir(txt_path)
    for filename in tqdm.tqdm(path_list):
        path = os.path.join(txt_path, filename)
        to_lf(path, isLF)


def xml2txt(class_names, xml_path, txt_path):
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


def visual_bbox(img_path, save_path, o_min, o_max):
    """
    :param img_path: path of image you want to draw bbox
    :param save_path: path of image that bboxes have been drawn
    :param o_min: truple(x_min, y_min)
    :param o_max: truple(x_max, y_max)
    """
    img = cv2.imread(img_path)

    cv2.rectangle(img, o_min, o_max, (0, 255, 0), 4)
    cv2.imwrite(save_path, img)


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

    print(boxesXXYY)

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


# data_config = parse_data_config("C:/Users/18917/Documents/Python Scripts/pytorch/Lab/PyTorch-YOLOv3-master/data/custom/classes.names")
# class_names = load_classes("C:/Users/18917/Documents/Python Scripts/pytorch/Lab/PyTorch-YOLOv3-master/data/custom/classes.names")
# print(class_names)
# xml2txt(class_names, 
#         "data/custom/weather/fog/xml",
#         "data/custom/weather/fog/labels")

# delete_null_files("data/custom/augmented/fog/labels",
#                   "data/custom/augmented/fog/images")

# create_dataset_txt("data/custom/weather/dusk/images", 
#                    "data/custom/weather/dusk/dusk_test.txt")


# class_names = load_classes("C:/Users/18917/Documents/Python Scripts/pytorch/Lab/PyTorch-YOLOv3-master/data/custom/classes.names")
# print(class_names)
# txt2xml(class_names, 
#         "data/custom/test/images", 
#         "data/custom/test/labels", 
#         "data/custom/test/xmls")

# getAddedData("data/custom/added.txt", "data/custom/added2.txt", "data/custom/shuffled/images/")

# pltBbox("data/custom/images/DSC_0057.jpg","data/custom/labels/DSC_0057.txt")

# print_txt_category("data/custom/test/labels")

# resize_images("C:/Users/18917/Documents/Python Scripts/pytorch/Lab/Pix2Pix-forlab/data-resize/6", 
#               "C:/Users/18917/Documents/Python Scripts/pytorch/Lab/Pix2Pix-forlab/data-square/6",
#               new_size=416)

# create_dataset_txt("C:/Users/18917/Documents/Python Scripts/pytorch/Lab/Pytorch-Utils.git/DataAugment/AutoWCT2/data/sunny", 
#                    "C:/Users/18917/Documents/Python Scripts/pytorch/Lab/Pytorch-Utils.git/DataAugment/AutoWCT2/data/train.txt")

# txt2xml(class_names, "C:/Users/18917/Documents/Python Scripts/pytorch/Lab/PyTorch-YOLOv3-master/data/custom/shuffled/images",
#         "C:/Users/18917/Documents/Python Scripts/pytorch/Lab/PyTorch-YOLOv3-master/data/custom/shuffled/labels", "C:/Users/18917/Documents/Python Scripts/pytorch/Lab/PyTorch-YOLOv3-master/data/custom/shuffled/labels")

# convert_to_single_class_txt("C:/Users/18917/Documents/Python Scripts/pytorch/Lab/PyTorch-YOLOv3-master/data/custom/shuffled/labels",
#                             "C:/Users/18917/Documents/Python Scripts/pytorch/Lab/PyTorch-YOLOv3-master/data/custom/single/labels")