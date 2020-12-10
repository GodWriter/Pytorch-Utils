import os
import cv2
import json
import tqdm

import numpy as np
import xml.etree.ElementTree as ET


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class voc2coco:
    def __init__(self, devkit_path=None, year=None):
        self.classes = ('__background__',  
                        'aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor')
 
        self.num_classes = len(self.classes)
        assert 'VOCdevkit' in devkit_path, 'VOC地址不存在: {}'.format(devkit_path)
        self.data_path = os.path.join(devkit_path, 'VOC' + year)
        self.annotaions_path = os.path.join(self.data_path, 'Annotations')
        self.image_set_path = os.path.join(self.data_path, 'ImageSets')
        self.year = year
        self.categories_to_ids_map = self._get_categories_to_ids_map()
        self.categories_msg = self._categories_msg_generator()
 
    def _load_annotation(self, ids=[]):
        ids = ids if _isArrayLike(ids) else [ids]
        image_msg = []
        annotation_msg = []
        annotation_id = 1
        for index in ids:
            filename = index
            print(filename)
            json_file = os.path.join(self.data_path, 'Segmentation_json', filename + '.json')
            if os.path.exists(json_file):
                img_file = os.path.join(self.data_path, 'JPEGImages', filename + '.jpg')
                im = cv2.imread(img_file)
                width = im.shape[1]
                height = im.shape[0]
                seg_data = json.load(open(json_file, 'r'))
                assert type(seg_data) == type(dict()), 'annotation file format {} not supported'.format(type(seg_data))
                for shape in seg_data['shapes']:
                    seg_msg = []
                    for point in shape['points']:
                        seg_msg += point
                    one_ann_msg = {"segmentation": [seg_msg],
                                   "area": self._area_computer(shape['points']),
                                   "iscrowd": 0,
                                   "image_id": int(index),
                                   "bbox": self._points_to_mbr(shape['points']),
                                   "category_id": self.categories_to_ids_map[shape['label']],
                                   "id": annotation_id,
                                   "ignore": 0
                                   }
                    annotation_msg.append(one_ann_msg)
                    annotation_id += 1
            else:
                xml_file = os.path.join(self.annotaions_path, filename + '.xml')
                tree = ET.parse(xml_file)
                size = tree.find('size')
                objs = tree.findall('object')
                width = size.find('width').text
                height = size.find('height').text
                for obj in objs:
                    bndbox = obj.find('bndbox')
                    [xmin, xmax, ymin, ymax] \
                        = [int(bndbox.find('xmin').text) - 1, int(bndbox.find('xmax').text),
                           int(bndbox.find('ymin').text) - 1, int(bndbox.find('ymax').text)]
                    if xmin < 0:
                        xmin = 0
                    if ymin < 0:
                        ymin = 0
                    bbox = [xmin, xmax, ymin, ymax]
                    print("image_id: ", index)
                    one_ann_msg = {"segmentation": self._bbox_to_mask(bbox),
                                   "area": self._bbox_area_computer(bbox),
                                   "iscrowd": 0,
                                   "image_id": int(index),
                                   "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                                   "category_id": self.categories_to_ids_map[obj.find('name').text],
                                   "id": annotation_id,
                                   "ignore": 0
                                   }
                    annotation_msg.append(one_ann_msg)
                    annotation_id += 1
            one_image_msg = {"file_name": filename + ".jpg",
                             "height": int(height),
                             "width": int(width),
                             "id": int(index)
                             }
            image_msg.append(one_image_msg)
        return image_msg, annotation_msg
    
    def _bbox_to_mask(self, bbox):
        assert len(bbox) == 4, 'Wrong bndbox!'
        mask = [bbox[0], bbox[2], bbox[0], bbox[3], bbox[1], bbox[3], bbox[1], bbox[2]]
        return [mask]
    
    def _bbox_area_computer(self, bbox):
        width = bbox[1] - bbox[0]
        height = bbox[3] - bbox[2]
        return width * height
    
    def _save_json_file(self, filename=None, data=None):
        json_path = os.path.join(self.data_path, 'cocoformatJson')
        assert filename is not None, 'lack filename'
        if os.path.exists(json_path) == False:
            os.mkdir(json_path)
        if not filename.endswith('.json'):
            filename += '.json'
        assert type(data) == type(dict()), 'data format {} not supported'.format(type(data))
        with open(os.path.join(json_path, filename), 'w') as f:
            f.write(json.dumps(data))
            
    def _get_categories_to_ids_map(self):
        return dict(zip(self.classes, range(self.num_classes)))
    
    def _get_all_indexs(self):
        ids = []
        for root, dirs, files in os.walk(self.annotaions_path, topdown=False):
            for f in files:
                if str(f).endswith('.xml'):
                    id = int(str(f).strip('.xml'))
                    ids.append(id)
        assert ids is not None, 'There is none xml file in {}'.format(self.annotaions_path)
        return ids
    
    def _get_indexs_by_image_set(self, image_set=None):
        if image_set is None:
            return self._get_all_indexs()
        else:
            image_set_path = os.path.join(self.image_set_path, 'Main', image_set + '.txt')
            assert os.path.exists(image_set_path), 'Path does not exist: {}'.format(image_set_path)
            with open(image_set_path) as f:
                ids = [x.strip() for x in f.readlines()]
            return ids
        
    def _points_to_mbr(self, points):
        assert _isArrayLike(points), 'Points should be array like!'
        x = [point[0] for point in points]
        y = [point[1] for point in points]
        assert len(x) == len(y), 'Wrong point quantity'
        xmin, xmax, ymin, ymax = min(x), max(x), min(y), max(y)
        height = ymax - ymin
        width = xmax - xmin
        return [xmin, ymin, width, height]
    
    def _categories_msg_generator(self):
        categories_msg = []
        for category in self.classes:
            if category == '__background__':
                continue
            one_categories_msg = {"supercategory": "none",
                                  "id": self.categories_to_ids_map[category],
                                  "name": category
                                  }
            categories_msg.append(one_categories_msg)
        return categories_msg
    
    def _area_computer(self, points):
        assert _isArrayLike(points), 'Points should be array like!'
        tmp_contour = []
        for point in points:
            tmp_contour.append([point])
        contour = np.array(tmp_contour, dtype=np.int32)
        area = cv2.contourArea(contour)
        return area
    
    def voc_to_coco_converter(self):
        img_sets = ['trainval', 'test']
        for img_set in img_sets:
            ids = self._get_indexs_by_image_set(img_set)
            img_msg, ann_msg = self._load_annotation(ids)
            result_json = {"images": img_msg,
                           "type": "instances",
                           "annotations": ann_msg,
                           "categories": self.categories_msg}
            self._save_json_file('voc_' + self.year + '_' + img_set, result_json)


def get_json_by_catID(json_path, save_path):
    """
    Get json file by id of category.
    1: liner 2: container 3: bulk 4: island 5: sailboat 6: other
    """
    fp = open(json_path, 'r', encoding='utf8')

    old_json = json.load(fp)
    new_json = {"images": [], 
                "type": old_json['type'], 
                "annotations": old_json['annotations'], 
                "categories": old_json['categories']}
    
    for ann in new_json["annotations"]:
        if ann['category_id'] == 2:
            new_json["images"].append(old_json["images"][ann['image_id']-1])
    
    with open(save_path, 'a', encoding='utf8')as fp:
        json.dump(new_json, fp, ensure_ascii=False)

    fp.close()


def compound_2_dataset(A_path, B_path, save_path):
    """
    Compose two json files into 1.
    """
    fp_A = open(A_path, 'r', encoding='utf8')
    fp_B = open(B_path, 'r', encoding='utf8')

    json_A = json.load(fp_A)
    json_B = json.load(fp_B)

    imgs = json_A["images"] + json_B["images"]
    anns = json_A["annotations"] + json_B["annotations"]

    new_json = {"images": imgs, 
                "type": json_A['type'], 
                "annotations": anns, 
                "categories": json_A['categories']}

    with open(save_path, 'a', encoding='utf8')as fp:
        json.dump(new_json, fp, ensure_ascii=False)

    fp_A.close()
    fp_B.close()


def rename_json_filename(json_path, save_path):
    fp = open(json_path, 'r', encoding='utf8')

    old_json = json.load(fp)
    new_json = {"images": [], 
                "type": old_json['type'], 
                "annotations": [], 
                "categories": old_json['categories']}
    
    for img in tqdm.tqdm(old_json["images"]):
        img['file_name'] = '1' + img['file_name']
        img['id'] += 2000
        new_json['images'].append(img)
    
    for ann in tqdm.tqdm(old_json["annotations"]):
        ann['image_id'] += 2000
        ann['id'] += 3000
        new_json['annotations'].append(ann)
    
    with open(save_path, 'a', encoding='utf8')as fp:
        json.dump(new_json, fp, ensure_ascii=False)

    fp.close()


def modify_json_value_type(json_path):
    fp = open(json_path, 'r', encoding='utf8')
    json_file = json.load(fp)

    for img in tqdm.tqdm(json_file["images"]):
        print(type(img['id']))
    
    for ann in tqdm.tqdm(json_file["annotations"]):
        print(ann(img['image_id']))


def rename_file(file_path):
    file_list = os.listdir(file_path)

    for name in tqdm.tqdm(file_list):
        old_path = os.path.join(file_path, name)
        new_path = os.path.join(file_path, '1' + name)

        os.rename(old_path, new_path)


# get_json_by_catID('config/coco/train.json', 'config/coco/container.json')
# rename_json_filename("C:/Users/18917/Documents/Python Scripts/pytorch/Lab/shipContest/data/b_test/b_test.json",
#                      "C:/Users/18917/Documents/Python Scripts/pytorch/Lab/shipContest/data/b_test/c_test.json")

# compound_2_dataset("C:/Users/18917/Documents/Python Scripts/pytorch/Lab/shipContest/data/b_test/a_test.json",
#                    "C:/Users/18917/Documents/Python Scripts/pytorch/Lab/shipContest/data/b_test/c_test.json",
#                    "C:/Users/18917/Documents/Python Scripts/pytorch/Lab/shipContest/data/b_test/d_test.json")

# rename_file("C:/Users/18917/Documents/Python Scripts/pytorch/Lab/shipContest/data/b_test/pic")

# modify_json_value_type("C:/Users/18917/Documents/Python Scripts/pytorch/Lab/shipContest/data/b_test/a_test.json")

# compound_2_dataset("C:/Users/18917/Documents/Python Scripts/pytorch/Lab/shipContest/data/sun_aug/train.json",
#                    "C:/Users/18917/Documents/Python Scripts/pytorch/Lab/shipContest/data/sun_aug/aug.json",
#                    "C:/Users/18917/Documents/Python Scripts/pytorch/Lab/shipContest/data/sun_aug/train_aug.json")

