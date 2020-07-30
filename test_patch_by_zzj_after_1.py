import sys
import time
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw
from utils import *
from load_data import PatchTransformer, PatchApplier, InriaDataset
import json
import numpy as np
import matplotlib.pyplot as plt



def compute_IOU(bbox1_o, bbox2_o):
    '''

    :param bbox1_o: list: up_left_x, up_left_y, width, height
    :param bbox2_o: list: up_left_x, up_left_y, width, height
    :return:
    '''
    # bbox1 = [0., 0., 0., 0.]
    # bbox2 = [0., 0., 0., 0.]
    # bbox1[0] = bbox1_o[0]
    bbox1 = bbox1_o.copy()
    bbox2 = bbox2_o.copy()
    bbox1[2] = bbox1_o[2] + bbox1_o[0]
    bbox1[3] = bbox1_o[3] + bbox1_o[1]
    bbox2[2] = bbox2_o[2] + bbox2_o[0]
    bbox2[3] = bbox2_o[3] + bbox2_o[1]

    ixmin = max(bbox1[0], bbox2[0])
    iymin = max(bbox1[1], bbox2[1])
    ixmax = min(bbox1[2], bbox2[2])
    iymax = min(bbox1[3], bbox2[3])
    iw = max(ixmax - ixmin, 0.)
    ih = max(iymax - iymin, 0.)
    inters = iw * ih

    # union
    uni = ((bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]) +
           (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]) - inters)

    overlap = inters / uni
    return overlap


def match_det(det_list, anno_list, threshold):

    for item in det_list:
        item['tp'] = 0
        item['fp'] = 0
        name = item['image_id']
        wait_anno_list = []
        overlap_list = []
        for item2 in anno_list:
            if item2['image_id'] == name:
                wait_anno_list.append(item2['bbox'])

        if len(wait_anno_list) > 0:
            anno_bbox_to_del_list = []
            for item3 in wait_anno_list:
                overlap = compute_IOU(item['bbox'], item3)
                overlap_list.append(overlap)
                anno_bbox_to_del_list.append(item3)
            ovmax = max(overlap_list)
            jmax = overlap_list.index(max(overlap_list))
            anno_bbox_to_del = anno_bbox_to_del_list[jmax]


            if ovmax > threshold:
                item['tp'] = 1
                for item4 in anno_list:
                    if item4['image_id'] == name and item4['bbox'] == anno_bbox_to_del:
                        # print(len(anno_list))
                        anno_list.remove(item4)
                        continue
                        # print(len(anno_list))
            else:
                item['fp'] = 1
        else:
            item['fp'] = 1
    return det_list


def get_score(elem):
    return elem['score']


def get_first(elem):
    return elem[0]


def get_second(elem):
    return elem[1]


def pr(det_list2, anno_list2, threshold):
    import copy
    det_list = copy.deepcopy(det_list2)
    anno_list = copy.deepcopy(anno_list2)
    GTBB_num = len(anno_list)
    det_list.sort(key=get_score, reverse=True)
    det_list_tp_fp = match_det(det_list, anno_list, threshold)
    det_list_tp_fp.sort(key=get_score, reverse=True)  # reverse=True : from big to small
    recall_list = []
    precision_list = []
    recall_precision_list = []
    confidence_old = 0

    list_show = []
    for it in det_list_tp_fp:
        list_show.append([it['image_id'], it['score'], it['tp']])

    for m in range(1, len(det_list_tp_fp)):
        tp = 0
        fp = 0
        tpsum = []

        if det_list_tp_fp[m]['tp'] == det_list_tp_fp[m]['fp']:
            print('error kkkkkkkkkkkkkkkk')
        for k in range(m):
            tp = tp + det_list_tp_fp[k]['tp']
            fp = fp + det_list_tp_fp[k]['fp']
        tpsum.append(tp)
        recall = tp/GTBB_num
        precision = tp/(tp+fp)


        if det_list_tp_fp[m]['score'] != confidence_old:
            recall_list.append(recall)
            precision_list.append(precision)
            recall_precision_list.append([recall, precision])
        else:
            # recall_list.pop()
            # precision_list.pop()
            # recall_precision_list.pop()
            recall_list.append(recall)
            precision_list.append(precision)
            recall_precision_list.append([recall, precision])
        confidence_old = det_list_tp_fp[m]['score']

        recall_list.append(recall)
        precision_list.append(precision)
        recall_precision_list.append([recall, precision])

    return recall_precision_list, recall_list, precision_list


def ap(recall_precision_list):
    if len(recall_precision_list) == 0:
        return float('nan')
    if len(recall_precision_list) == 1:
        repr = recall_precision_list[0]
        return repr[0] * repr[1]

    # use recall to sort , get_first
    recall_precision_list.sort(key=get_first, reverse=False)  # reverse=True : from big to small

    recall_list = []
    precision_list = []
    for item in recall_precision_list:
        recall_list.append(item[0])
        precision_list.append(item[1])

    recall_nparray = np.array(recall_list)
    precision_nparray = np.array(precision_list)
    d_recall = np.diff(recall_nparray, n=1, axis=-1)
    first_d_recall = np.array(recall_list[0])
    d_recall = np.append(first_d_recall, d_recall)
    ap = np.dot(d_recall, precision_nparray)

    return ap

if __name__ == '__main__':
    # read json
    import copy
    with open("clean_results.json", 'r') as load_f:
        clean_results = json.load(load_f)
    with open("noise_results.json", 'r') as load_f:
        noise_results = json.load(load_f)
    with open("patch_results1.json", 'r') as load_f:
        v4_result = json.load(load_f)
    threshold = 0.5

    plt.plot([0, 1.05], [0, 1.05], '--', color='gray')



    recall_precision_list_patch, recall_list_patch, precision_list_patch = pr(clean_results, clean_results, threshold=threshold)
    recall_nparray_patch = np.array(recall_list_patch)
    precision_nparray_patch = np.array(precision_list_patch)
    plt.plot(recall_nparray_patch, precision_nparray_patch)



    recall_precision_list_noise, recall_list_noise, precision_list_noise = pr(noise_results, clean_results, threshold=threshold)
    recall_nparray_noise = np.array(recall_list_noise)
    precision_nparray_noise = np.array(precision_list_noise)
    plt.plot(recall_nparray_noise, precision_nparray_noise)


    # recall_precision_list_patch, recall_list_patch, precision_list_patch = pr(class_shift, clean_results, threshold=threshold)
    # recall_nparray_patch = np.array(recall_list_patch)
    # precision_nparray_patch = np.array(precision_list_patch)
    # plt.plot(recall_nparray_patch, precision_nparray_patch)



    # recall_precision_list_patch, recall_list_patch, precision_list_patch = pr(up_results, clean_results, threshold=threshold)
    # recall_nparray_patch = np.array(recall_list_patch)
    # precision_nparray_patch = np.array(precision_list_patch)
    # plt.plot(recall_nparray_patch, precision_nparray_patch)

    # recall_precision_list_patch, recall_list_patch, precision_list_patch = pr(class_only, clean_results, threshold=threshold)
    # recall_nparray_patch = np.array(recall_list_patch)
    # precision_nparray_patch = np.array(precision_list_patch)
    # plt.plot(recall_nparray_patch, precision_nparray_patch)

    recall_precision_list_patch, recall_list_patch, precision_list_patch = pr(v4_result, clean_results, threshold=threshold)
    recall_nparray_patch = np.array(recall_list_patch)
    precision_nparray_patch = np.array(precision_list_patch)
    plt.plot(recall_nparray_patch, precision_nparray_patch)

    plt.gca().set_ylabel('Precision')
    plt.gca().set_xlabel('Recall')
    plt.gca().set_xlim([0, 1.05])
    plt.gca().set_ylim([0, 1.05])
    plt.gca().legend(loc=4)


    plt.show()
    print()
    # ap = ap(recall_precision_list)


