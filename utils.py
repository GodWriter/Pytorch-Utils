import os
import xml.etree.ElementTree as ET

from xml.dom.minidom import Document


def load_classes(path):
    fp = open(path, 'r')
    names = fp.read().split('\n')[:-1]
    return names


def convert_txt_format(txt_path, isLF):
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
    def getElement(ele, str_list):
        element_list = []
        for text in str_list:
            element = ele.find(text)
            element_list.append(float(element.text))
        
        return element_list
    
    xml_file = os.listdir(xml_path)
    for xml in xml_file:
        xml_ = os.path.join(xml_path, xml)
        tree = ET.parse(xml_)
        root = tree.getroot()

        # obtain height and width
        size = root.find("size")
        height = float(size.find("height").text)
        width = float(size.find("width").text)

        # obtain the bbox info
        for object in tree.iter(tag="object"):
            info = []
            name = object.find("name").text
            print(class_names.index(name))

            for bbox in object.iter(tag="bndbox"):
                element_list = getElement(bbox, ["xmin", "xmax", "ymin", "ymax"])
                print(element_list)
            



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
# print(class_names)
# xml2txt("data/ships/xmls", "")