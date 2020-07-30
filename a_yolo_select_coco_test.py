from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import torch
import numpy as np
import os
from tool.darknet2pytorch import *
from utils.utils import *
import json
# MAX_TOTAL_AREA_RATE = 0.12
# MIN_SPLIT_AREA_RATE = 0.0016
# selected_path = 'random_selected_img_800'
def count_score_yolov4(max_total_area_rate, min_split_area_rate, selected_path, max_patch_number, json_name):
    patch_temp_size = 800
    cfgfile = "cfg/yolov4.cfg"
    weightfile = "yolov4.weights"
    darknet_model = Darknet(cfgfile)
    darknet_model.load_weights(weightfile)
    darknet_model = darknet_model.eval().cuda()
    num_classes = 80
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    files = os.listdir(selected_path)
    resize2 = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Resize((patch_temp_size, patch_temp_size)),
        transforms.ToTensor()])
    files.sort()



    resize_small = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((608, 608)),
        # transforms.ToTensor()
    ])
    resize_small_500 = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((500, 500)),
        # transforms.ToTensor()
    ])
    save_path = 'select_from_test_500'
    from tqdm import tqdm
    ia = 0
    for img_index in tqdm(range(len(files))):
        if ia > 5000:
            break
        img_name = files[img_index]
        img_path0 = os.path.join(selected_path, img_name)
        img0 = Image.open(img_path0).convert('RGB')


        # print('-----------------')
        # print('Now testing', img_name)


        img0_608 = resize_small(img0)


        # --------------------BOX score
        boxes0 = do_detect(darknet_model, img0_608, 0.5, 0.4, True)
        # class_names = load_class_names(namesfile)
        # plot_boxes(img0, boxes0, 'predictions.jpg', class_names)
        box0_tensor = torch.tensor(boxes0)
        if len(boxes0) == 0:
            continue
        area = torch.mul(box0_tensor[:, 2], box0_tensor[:, 3])
        if torch.min(area) < 0.05:
            continue
        if 2 <= box0_tensor.shape[0] <= 15:
            img0_500 = resize_small_500(img0)
            item = str(ia)+'.png'
            img0_500.save(os.path.join(save_path, item))
            print(ia)
            ia += 1


        # area = torch.mul((boxes0[:, 2] - boxes0[:, 0]), (boxes0[:, 3] - boxes0[:, 1]))
        # if torch.min(area / (h * w)) < 0.05:
        #     continue
        # if 4 <= gt_bboxes.size()[0] <= 15:










if __name__ == '__main__':
    MAX_TOTAL_AREA_RATE = 0.12
    MIN_SPLIT_AREA_RATE = 0.0016
    selected_path = '/disk2/mycode/0511models/coco/test2017'
    max_patch_number = 10
    json_name = 'bbox_score.json'

    count_score_yolov4(MAX_TOTAL_AREA_RATE, MIN_SPLIT_AREA_RATE, selected_path, max_patch_number, json_name)
    print('total socre is')