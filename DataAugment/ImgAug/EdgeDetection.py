import os
import cv2
import tqdm


def inverse_color(image):
    height, width = image.shape
    img2 = image.copy()

    for i in range(height):
        for j in range(width):
            img2[i, j] = (255 - image[i, j])

    return img2


def canny_edge_detection(img_path):
    img = cv2.imread(img_path, 0)
    blur = cv2.GaussianBlur(img, (3, 3), 0)

    canny = cv2.Canny(blur, 50, 150)
    canny_inverse = inverse_color(canny)

    return canny_inverse


IMG_PATH = "C:/Users/18917/Documents/Python Scripts/pytorch/Lab/PyTorch-YOLOv3-master/data/custom/shuffled/resized416x416/images"
SAVE_PATH = "C:/Users/18917/Documents/Python Scripts/pytorch/Lab/PyTorch-YOLOv3-master/data/custom/shuffled/resized416x416/edge"


name_list = os.listdir(IMG_PATH)
for name in tqdm.tqdm(name_list):
    img_path = os.path.join(IMG_PATH, name)
    save_img = os.path.join(SAVE_PATH, name)

    canny = canny_edge_detection(img_path)
    cv2.imwrite(save_img, canny)