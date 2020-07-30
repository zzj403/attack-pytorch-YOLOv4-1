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


    connected_domin_json_name = 'connected_domin_score_dict.json'
    with open(connected_domin_json_name) as f_obj:
        connected_domin_score_dict = json.load(f_obj)
    bb_score_dict = {}
    for img_name in files:
        # print(k,v)




        img_path0 = os.path.join(selected_path, img_name)
        img0 = Image.open(img_path0).convert('RGB')


        # print('-----------------')
        # print('Now testing', img_name)

        resize_small = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((608, 608)),
            # transforms.ToTensor()
        ])
        img0 = resize_small(img0)


        # --------------------BOX score
        boxes0 = do_detect(darknet_model, img0, 0.5, 0.4, True)


        class_names = load_class_names(namesfile)
        # plot_boxes(img1, boxes1, 'predictions.jpg', class_names)
        if len(boxes0) == 1:
            print('Fatal ERROR: YOLOv4 can only find 1 thing in the clean image', img_name)
            os.remove(os.path.join(selected_path, img_name))

        if len(boxes0) == 2:
            print('Fatal ERROR: YOLOv4 can only find 2 thing in the clean image', img_name)
            os.remove(os.path.join(selected_path, img_name))


        if len(boxes0) == 0:
            print('Fatal ERROR: YOLOv4 can\'t find anything in the clean image', img_name)
            os.remove(os.path.join(selected_path, img_name))
            # assert len(boxes0) != 0
            bb_score = 0








if __name__ == '__main__':
    MAX_TOTAL_AREA_RATE = 0.12
    MIN_SPLIT_AREA_RATE = 0.0016
    selected_path = 'select2000_new'
    max_patch_number = 10
    json_name = 'bbox_score.json'

    count_score_yolov4(MAX_TOTAL_AREA_RATE, MIN_SPLIT_AREA_RATE, selected_path, max_patch_number, json_name)
    print('total socre is')