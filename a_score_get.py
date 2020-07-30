import numpy as np

# b = np.loadtxt('connected_domin_score_list.csv', delimiter=',' )
# print(b)


import json
from a_connect_region_detect import *
from infer import *


json_name = 'connected_domin_score_dict.json'

# count_score_yolov4(MAX_TOTAL_AREA_RATE, MIN_SPLIT_AREA_RATE, selected_path, max_patch_number, json_name)


with open(json_name) as f_obj:
    connected_domin_score_dict = json.load(f_obj)

# print("numbers = ", connected_domin_score_dict)


'/disk2/mycode/0511models/mmdetection-master/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py /disk2/mycode/0511models/mmdetection-master/checkpoints/faster_rcnn/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth ./select1000_p/'

# read 2 json get score
bbox_score_json_file = 'bbox_score.json'
with open(bbox_score_json_file) as f_obj:
    bbox_score_dict = json.load(f_obj)
assert len(bbox_score_dict) == len(connected_domin_score_dict)
# bbox_score_dict
# connected_domin_score_dict
score = 0
for (k, v) in bbox_score_dict.items():
    # print('-------------------')
    # print(k)
    # print(connected_domin_score_dict[k])
    print(bbox_score_dict[k])
    score += connected_domin_score_dict[k] * bbox_score_dict[k]

print('score', score)

