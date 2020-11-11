import os
import cv2
import json
import shutil

import numpy as np
import xml.etree.ElementTree as ET

from PIL import Image
from to_coco import voc2coco
from xml.dom.minidom import Document


class voctools():
    def __init__(self):
        pass
    
    @staticmethod
    def convert_txt_format(txt_path, isLF):
        """
        :param txt_path: path of all txt files
        :param isLF: True means converting to Unix(LF), False means converting to Windows(CRLF)
        """
        # function converting file format and encoding.
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
    
    @staticmethod
    def remove_image_labelled(xml_path, data_path, data_labelled_path, image_format='.png'):
        """
        :param xml_path: path of the xml
        :param data_path: path of all the images
        :param data_labelled_path: path of all the images labelled
        :param image_format: suffix of the images
        """
        xml_file = os.listdir(xml_path)
        image_unlabel = os.listdir(dataset_path)
        image_labelled = os.listdir(dataset_labelled_path)

        # delete the image suffix
        img_name_list = []
        for img in image_unlabel:
            img_name_list.append(img[:-4])
        
        # delete the xml suffix
        xml_name_list = []
        for xml in xml_file:
            xml_name_list.append(xml[:-4])
        
        for xml in xml_name_list:
            if xml in img_name_list:
                old_dir = dataset_path + xml + image_format
                new_dir = dataset_labelled_path + xml + image_format
                shutil.copyfile(old_dir, new_dir)
                os.remove(old_dir)
    
    @staticmethod
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
    
    @staticmethod
    def create_xml(bbox_list, save_path):
        """
        :param bbox_list: list of bouning box 
        """
        def createNode(name, value):
            node = doc.createElement(name)
            if value is not None:
                node_text = doc.createTextNode(value)
                node.appendChild(node_text)

            return node
        
        doc = Document()
        annotation = doc.createElement("annotation")

        annotation.appendChild(createNode("folder", "voc2012"))
        annotation.appendChild(createNode("filename", "111.jpg"))
        annotation.appendChild(createNode("source", "None"))

        # create node of size
        size = createNode("size", None)
        size.appendChild(createNode("width", str(10)))
        size.appendChild(createNode("height", str(10)))
        size.appendChild(createNode("depth", str(3)))
        annotation.appendChild(size)

        for box in bbox_list:
            object_ = createNode("object", None)
            object_.appendChild(createNode("name", "boat"))
            object_.appendChild(createNode("pose", "Unspecified"))
            object_.appendChild(createNode("truncated", str(0)))
            object_.appendChild(createNode("difficult", str(0)))
            
            # create bounding box
            bndbox = createNode("bndbox", None)
            bndbox.appendChild(createNode("xmin", str(box[0])))
            bndbox.appendChild(createNode("ymin", str(box[1])))
            bndbox.appendChild(createNode("xmax", str(box[2])))
            bndbox.appendChild(createNode("ymax", str(box[3])))
            object_.appendChild(bndbox)
            
            annotation.appendChild(object_)

        doc.appendChild(annotation)

        with open(save_path, 'wb') as f:
            f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
    
    @staticmethod
    def voc_to_coco(coco_path, year):
        """
        :param coco_path: pascal path to be converted, default is './VOC2012/VOCdevkit'
        :param year: '2012', '2007', it should copy with coco_path
        """
        converter = voc2coco(coco_path, year)
        converter.voc_to_coco_converter()
    
    @staticmethod
    def resize_xml_bbox(xml_path, shrink_num):
        """
        :param xml_path: name list of xml files
        :param shrink_num: how big you want to shrink the bbox, etc. 2, 3, 4 ...
        """
        def get_element(ele, str_list):
            element_list = []
            for text in str_list:
                element = ele.find(text)
                element_list.append(element)
            
            return element_list

        def do_operation(element_list, shrink_num):
            for element in element_list:
                value = int(int(element.text) / shrink_num)
                element.text = str(value)
        
        xml_file = os.listdir(xml_path)
        for xml in xml_file:
            xml_ = os.path.join(xml_path, xml)
            tree = ET.parse(xml_)
            
            # modify the bbox
            for elem in tree.iter(tag='bndbox'):
                element_list = get_element(elem, ["xmin", "xmax", "ymin", "ymax"])
                do_operation(element_list, shrink_num)

            # modify the width and height
            for elem in tree.iter(tag='height'):
                value = int(int(elem.text) / shrink_num)
                elem.text = str(value)

            for elem in tree.iter(tag='width'):
                value = int(int(elem.text) / shrink_num)
                elem.text = str(value)
            
            tree.write(xml_)
    
    @staticmethod
    def resize_image(img_path, save_path, shrink_num):
        """
        :param img_path: name list of images
        :param save_path: path to save the image resized
        :param shrink_num: how big you want to shrink the image, etc. 2, 3, 4 ...
        """
        for name in os.listdir(img_path):
            old_path = os.path.join(img_path, name)
            new_path = os.path.join(save_path, name)
            
            img = Image.open(old_path)
            img = img.convert('RGB')

            # width, height = int(img.size[0]/shrink_num), int(img.size[1]/shrink_num)
            img = img.resize((96, 96), Image.ANTIALIAS)
            
            img.save(new_path)
    
    @staticmethod
    def resize_image_with_bbox(img_path, xml_path, img_save_path, shrink_num):
        """
        :param img_path: name list of images
        :param xml_path: name list of xml files
        :param img_save_path: path to save the image resized
        :param shrink_num: how big you want to shrink the image, etc. 2, 3, 4 ...
        """
        voctools.resize_image(img_path, img_save_path, shrink_num)
        print("images have been resized!")

        voctools.resize_xml_bbox(xml_path, shrink_num)
        print("xmls have been modified!")
    
    @staticmethod
    def convert_image_format(img_path, save_path, i_format):
        """
        :param img_path: name list of images
        :param save_path: path to save the image converted
        :param i_format: if 'png' means from 'jpg' to 'png', vice versa
        """
        img_list = os.listdir(img_path)

        for name in img_list:
            old_path = os.path.join(img_path, name)

            if i_format == 'jpg':
                new_path = os.pardir.join(save_path, name[:-4] + '.jpg')
            elif i_format == 'png':
                new_path = os.pardir.join(save_path, name[:-4] + '.png')

            img = Image.open(old_path)
            img = img.convert('RGB')
            img.save(new_path)
    
    @staticmethod
    def modify_xml_node_text(xml_path):
        """
        :param xml_path: name list of xml files
        """
        xml_name = os.listdir(xml_path)

        for name in xml_name:
            xml = os.path.join(xml_path, name)

            tree = ET.parse(xml)
            root = tree.getroot()
            element_filename = root.find("filename")
            element_filename.text = 'sunset' + element_filename.text

            tree.write(xml)
    
    @staticmethod
    def create_filename_txt(file_path, save_path):
        """
        :param file_path: path of the files
        :param save_path: path of the txt saved
        """
        file_list = os.listdir(file_path)
        saved_txt = open(save_path, 'a+')

        for filename in file_list:
            content = filename[:-4] + '\n'
            saved_txt.write(content)
        
        saved_txt.close()

    @staticmethod
    def crop_xml_bbox(xml_path, number):
        """
        :param xml_path: name list of xml files
        :param shrink_num: how big you want to shrink the bbox, etc. 2, 3, 4 ...
        """
        def get_element(ele, str_list):
            element_list = []
            for text in str_list:
                element = ele.find(text)
                element_list.append(element)
            
            return element_list

        def do_operation(element_list, shrink_num):
            for element in element_list:
                value = int(int(element.text) / shrink_num)
                element.text = str(value)
        
        xml_file = os.listdir(xml_path)
        for xml in xml_file:
            xml_ = os.path.join(xml_path, xml)
            tree = ET.parse(xml_)
            
            # modify the bbox
            for elem in tree.iter(tag='bndbox'):
                element_list = get_element(elem, ["xmin", "xmax", "ymin", "ymax"])
                do_operation(element_list, shrink_num)

            # modify the width and height
            for elem in tree.iter(tag='height'):
                value = int(int(elem.text) / shrink_num)
                elem.text = str(value)

            for elem in tree.iter(tag='width'):
                value = int(int(elem.text) / shrink_num)
                elem.text = str(value)
            
            tree.write(xml_)
    
    @staticmethod
    def get_txt_lines(txt_path):
        """
        :param txt_path: path of the txt file
        """
        name_list = []

        with open(txt_path, 'r') as f:
            line = f.readline()
            while line:
                name_list.append(line.split()[0])
                line = f.readline()
        f.close()
        
        return name_list
    
    @staticmethod
    def modify_bbox_height(txt_path, xml_path, save_path, number):
        """
        :param xml_path: name list of xml files
        """
        def get_element(ele, str_list):
            element_list = []
            for text in str_list:
                element = ele.find(text)
                element_list.append(element)

            return element_list

        def do_operation(element_list, shrink_num):
            for element in element_list:
                value = int(int(element.text) / shrink_num)
                element.text = str(value)

        xml_name = get_txt_lines(txt_path)
        for xml in xml_name:
            xml_old = os.path.join(xml_path, xml + '.xml')
            xml_new = os.path.join(save_path, 'down' + xml + '.xml')
            
            tree = ET.parse(xml_old)

            # modify the width and height
            for elem in tree.iter(tag='height'):
                value = int(number)
                elem.text = str(value)

            tree.write(xml_new)
        
        print("Have been finished !")
    
    @staticmethod
    def modify_bbox_and_height(txt_path, xml_path, save_path, number):
        """
        :param xml_path: name list of xml files
        """
        def get_element(ele, str_list):
            element_list = []
            for text in str_list:
                element = ele.find(text)
                element_list.append(element)

            return element_list

        def do_operation(element_list, number):
            for element in element_list:
                value = int(int(element.text) - number)
                element.text = str(value)

        xml_name = get_txt_lines(txt_path)
        for xml in xml_name:
            xml_old = os.path.join(xml_path, xml + '.xml')
            xml_new = os.path.join(save_path, 'top' + xml + '.xml')
            
            tree = ET.parse(xml_old)

            # modify the bbox
            for elem in tree.iter(tag='bndbox'):
                element_list = get_element(elem, ["ymin", "ymax"])
                do_operation(element_list, number)

            # modify the width and height
            for elem in tree.iter(tag='height'):
                value = int(int(elem.text) - number)
                elem.text = str(value)

            tree.write(xml_new)
        
        print("Have been finished !")
    
    @staticmethod
    def rename_file(file_path):
        """
        :param file_path: name list of files
        """
        file_names = os.listdir(file_path)

        for name in file_names:
            old_name = os.path.join(file_path, name)
            new_name = os.path.join(file_path, 'sunset' + name)

            os.rename(old_name, new_name)
    
    @staticmethod
    def modify_txt_file(file_path, save_path, text):
        """
        :param file_path: name list of files
        """
        source_file = open(file_path)

        for line in source_file:
            line = line.strip('\n')
            line = text + line
            line = line + '\n'

            with open(save_path, 'a+') as f:
                f.write(line)

        source_file.close()
    
    @staticmethod
    def get_class_file(file_path, save_path):
        """
        :param file_path: txt file of images
        :param save_path: txt file to save
        """
        class_id = []

        with open(file_path, "r") as f:
            for line in f:
                line_ = line.split()
                
                if line_[1] == "1":
                    class_id.append(line_[0])
        
        with open(save_path, 'w') as f:
            for idx in class_id:
                f.write(idx + '\n')
    
    @staticmethod
    def find_file_by_id(id_file, old_path, new_path, old_format, new_format):
        """
        :param id_file: file that contains id list
        :param old_path: path to find file
        :param new_path: path to save file
        :param save_path: directory to save files
        """
        id_list = []

        with open(id_file, "r") as f:
            for line in f:
                line_ = line.split()
                id_list.append(line_[0])
        
        for id in id_list:
            old_dir = old_path + '/' + id + old_format
            new_dir = new_path + '/' + id + new_format
            shutil.copyfile(old_dir, new_dir)