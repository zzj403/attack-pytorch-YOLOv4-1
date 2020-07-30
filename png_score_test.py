from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import torch
import numpy as np
import os
from tool.darknet2pytorch import *
from utils.utils import *

# MAX_TOTAL_AREA_RATE = 0.12
# MIN_SPLIT_AREA_RATE = 0.0016
# selected_path = 'random_selected_img_800'
def count_score_yolov4(max_total_area_rate, min_split_area_rate, selected_path, max_patch_number):
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

    bb_score_list = []
    bb_0_list = []
    bb_1_list = []
    total_area_rate_list = []
    connected_domin_score_list = []
    for img_name_index in range(len(files)):

        img_name = files[img_name_index]
        if img_name_index > 100:
            break

        img_path0 = os.path.join(selected_path, img_name)
        img0 = Image.open(img_path0).convert('RGB')
        img_path1 = os.path.join(selected_path+'_p', img_name.split('.')[0]+'_p.png')
        img1 = Image.open(img_path1).convert('RGB')
        img0_t = resize2(img0).cuda()
        img1_t = resize2(img1).cuda()
        img_minus_t = img0_t - img1_t

        # img = transforms.ToPILImage()(img_minus_t.detach().cpu())
        # img.show()
        print('-----------------')
        print('Now testing', img_name)
        connected_domin_score, total_area_rate, patch_number = \
            connected_domin_detect_and_score(img_minus_t, max_total_area_rate, min_split_area_rate, max_patch_number)

        if patch_number > max_patch_number:
            print(img_name, '\'s patch number is too many =', str(patch_number), ' Its score will not be calculated')
            print('Required patch number is', str(max_patch_number))
            continue

        if total_area_rate > max_total_area_rate:
            print(img_name, '\'s patch is too large =', str(total_area_rate), ' Its score will not be calculated')
            print('Required patch area rate is', str(max_total_area_rate))
            continue

        print('area score is', str(float(connected_domin_score)))
        # print('total_area_rate is', str(total_area_rate))
        total_area_rate_list.append(total_area_rate)
        connected_domin_score_list.append(connected_domin_score)


        # --------------------BOX score
        boxes0 = do_detect(darknet_model, img0, 0.5, 0.4, True)
        boxes1 = do_detect(darknet_model, img1, 0.5, 0.4, True)
        np.savetxt("exaple.txt", boxes0)

        if len(boxes0) == 0:
            print('Fatal ERROR: YOLOv4 can\'t find anything in the clean image')
            # assert len(boxes0) != 0
        else:
            bb_score = 1 - min(len(boxes0), len(boxes1))/len(boxes0)
            print('bb score is', str(bb_score))
            bb_score_list.append(bb_score)
            bb_0_list.append(len(boxes0))
            bb_1_list.append(len(boxes1))


        class_names = load_class_names(namesfile)
        img0_b = plot_boxes(img0, boxes0, savename=None, class_names=class_names)
        img1_b = plot_boxes(img1, boxes1, savename=None, class_names=class_names)
        # img0_b.show()
        if img_name == '000000017743.png':
            img1_b.show()
            img = transforms.ToPILImage()(img_minus_t.detach().cpu())
            img.show()

    sum_score = 0
    score_list = []
    for i in range(len(bb_score_list)):
        # print(bb_1_list[i]/bb_0_list[i])
        score_now = bb_score_list[i]*connected_domin_score_list[i]
        score_list.append(score_now)
        sum_score = sum_score + score_now
    return sum_score



def connected_domin_detect_and_score(input_img, max_total_area_rate, min_split_area_rate, max_patch_number):
    from skimage import measure
    # detection
    input_img_new = (input_img[0]+input_img[1]+input_img[2])
    ones = torch.cuda.FloatTensor(input_img_new.size()).fill_(1)
    zeros = torch.cuda.FloatTensor(input_img_new.size()).fill_(0)

    whole_size = input_img_new.shape[0]*input_img_new.shape[1]
    input_map_new = torch.where((input_img_new != 0), ones, zeros)


    # img = transforms.ToPILImage()(input_map_new.detach().cpu())
    # img.show()
    input_map_new = input_map_new.cpu()
    labels = measure.label(input_map_new[:, :], background=0, connectivity=2)
    # print(labels)
    label_max_number = np.max(labels)
    if max_patch_number > 0:
        if label_max_number > max_patch_number:
            return 0, 0, float(label_max_number)


    total_area = 0
    for i in range(1, label_max_number+1):
        label_map = torch.from_numpy(labels).cuda()
        now_count_map = torch.where((label_map == i), ones, zeros)
        now_count_area = now_count_map.sum()
        now_count_area_rate = now_count_area / whole_size
        if now_count_area_rate < min_split_area_rate:
            print('WARNING: A connected area rate ', float(now_count_area_rate), 'is smaller than', min_split_area_rate,
                  'as we required. max(limit, area) is used.')
        total_area = total_area + max(now_count_area_rate*whole_size, min_split_area_rate*whole_size)
    total_area_rate = total_area/whole_size
    # if total_area_rate >= max_total_area_rate:
    #     print('ERROR:Too large patch area at ', str(float(total_area_rate)), '! Required area is', str(max_total_area_rate))
    area_score = 1/total_area_rate
    return float(area_score), float(total_area_rate), float(label_max_number)





if __name__ == '__main__':
    MAX_TOTAL_AREA_RATE = 0.12
    MIN_SPLIT_AREA_RATE = 0.0016
    selected_path = 'random_selected_img_800'
    max_patch_number = 8


    x = count_score_yolov4(MAX_TOTAL_AREA_RATE, MIN_SPLIT_AREA_RATE, selected_path, max_patch_number)
    print('total socre is', x)