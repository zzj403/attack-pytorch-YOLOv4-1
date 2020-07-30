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
from load_data import *

def count_score_yolov4(selected_path, json_name):
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
    zero_num = 0
    for img_name, v in connected_domin_score_dict.items():
        # print(k,v)
        # if v == 0:
        #     bb_score_dict[img_name] = 0.0
        #     continue

        img_path0 = os.path.join('select_from_test_500_0615', img_name)
        img0 = Image.open(img_path0).convert('RGB')

        img_path1 = os.path.join(selected_path, img_name)
        img1 = Image.open(img_path1).convert('RGB')


        print('-----------------')
        print('Now testing', img_name)

        resize_small = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((608, 608)),
            # transforms.ToTensor()
        ])
        img0 = resize_small(img0)
        img1 = resize_small(img1)

        patched_img = F.interpolate(transforms.ToTensor()(img1).unsqueeze(0),
                                    (darknet_model.height, darknet_model.width),
                                    mode='bilinear').cuda()
        output = darknet_model(patched_img)

        prob_extractor = MaxProbExtractor(0, 80, 'ReproducePaperObj').cuda()
        max_prob = prob_extractor(output)


        # --------------------BOX score
        boxes0 = do_detect(darknet_model, img0, 0.5, 0.4, True)
        boxes1 = do_detect(darknet_model, img1, 0.5, 0.4, True)





        class_names = load_class_names(namesfile)
        plot_boxes(img1, boxes1, 'predictions.jpg', class_names)

        if len(boxes0) == 0:
            print('Fatal ERROR: YOLOv4 can\'t find anything in the clean image', img_name)
            # assert len(boxes0) != 0
            bb_score = 0
        else:
            bb_score = 1 - min(len(boxes0), len(boxes1))/len(boxes0)
            print('bb score is', str(bb_score))
        bb_score_dict[img_name] = bb_score
        if bb_score == 0:
            zero_num +=1
            print('zero_num:',zero_num)


    with open(json_name, 'w') as f_obj:
        json.dump(bb_score_dict, f_obj)







if __name__ == '__main__':

    selected_path = 'black_cutout'

    json_name = 'bbox_score.json'

    count_score_yolov4(selected_path, json_name)
    print('total socre is')