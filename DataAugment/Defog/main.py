import os
import cv2
import tqdm

from autoColorEqual import zmIceColor


if __name__ == "__main__":
    file_path = "C:/Users/18917/Documents/Python Scripts/pytorch/Lab/PyTorch-YOLOv3-master/data/custom/weather/fog/images"
    save_path = "C:/Users/18917/Documents/Python Scripts/pytorch/Lab/PyTorch-YOLOv3-master/data/custom/weather/defog/images"

    name_list = os.listdir(file_path)
    for name in tqdm.tqdm(name_list):
        old_path = os.path.join(file_path, name)
        new_path = os.path.join(save_path, name)

        img = zmIceColor(cv2.imread(old_path) / 255.0) * 255
        cv2.imwrite(new_path, img)