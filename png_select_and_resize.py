from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import torch
import numpy as np
import random

import os

IMAGE_SIZE = 800
SAVE_DIR = 'random_selected_img_800'


def main():
    coco_test_path = '/disk2/mycode/0511models/coco/test2017'
    files = os.listdir(coco_test_path)
    total_len = len(files)
    index_list = random_int_list(0, total_len - 1, 400)

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    for file_index in index_list:
        file_name = files[file_index]
        img_path = os.path.abspath(os.path.join(coco_test_path, file_name))
        _temp_img = Image.open(img_path).convert('RGB')
        resize2 = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()])
        img_t = resize2(_temp_img).cuda()
        img_pil = transforms.ToPILImage('RGB')(img_t.cpu())

        name = file_name.split(sep='.')[0]
        resized_img_name = name + ".png"
        img_pil.save(os.path.join(SAVE_DIR, resized_img_name))



    '/disk2/mycode/0511models/coco/test2017'


def random_int_list(start, stop, length):
    random.seed(10)
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        temp = random.randint(start, stop)
        while random_list.count(temp) != 0:
            temp = random.randint(start, stop)
        random_list.append(temp)
    return random_list


if __name__ == '__main__':
    main()
