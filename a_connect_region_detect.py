from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from utils.utils import *
import json
# MAX_TOTAL_AREA_RATE = 0.12
# MIN_SPLIT_AREA_RATE = 0.0016
# selected_path = 'random_selected_img_800'
def count_score_yolov4(max_total_area_rate, min_split_area_rate, selected_path, max_patch_number, json_name):
    cfgfile = "cfg/yolov4.cfg"
    weightfile = "yolov4.weights"

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
    connected_domin_score_dict = {}
    for img_name_index in range(len(files)):

        img_name = files[img_name_index]
        # if img_name_index > 100:
        #     break


        img_path0 = os.path.join(selected_path.replace('_p', ''), img_name)
        img0 = Image.open(img_path0).convert('RGB')
        img_path1 = os.path.join(selected_path, img_name)
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
            os.remove(os.path.join(selected_path, img_name))
            continue

        if patch_number == 0:
            print(img_name, '\'s patch number is 0. Its score will not be calculated')
            os.remove(os.path.join(selected_path, img_name))
            continue

        if total_area_rate > max_total_area_rate:
            print(img_name, '\'s patch is too large =', str(total_area_rate), ' Its score will not be calculated')
            print('Required patch area rate is', str(max_total_area_rate))
            os.remove(os.path.join(selected_path, img_name))
            continue

        # print('area score is', str(float(connected_domin_score)))
        # print('total_area_rate is', str(total_area_rate))
        total_area_rate_list.append(total_area_rate)
        connected_domin_score_dict[img_name] = connected_domin_score





    with open(json_name, 'w') as f_obj:
        json.dump(connected_domin_score_dict, f_obj)
    # b = np.loadtxt('connected_domin_score_list.csv', delimiter=',' )
    print()



    return connected_domin_score_dict



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
    if label_max_number == 0:
        return 0, 0, 0


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
    area_score = min(4, float(max_total_area_rate/total_area_rate))
    return float(area_score), float(total_area_rate), float(label_max_number)





if __name__ == '__main__':
    MAX_TOTAL_AREA_RATE = 0.02  # 20000/(1000*1000) = 0.02
    MIN_SPLIT_AREA_RATE = 0.001  # 1000/(1000*1000) = 0.001
    selected_path = 'select1000_new_p'
    max_patch_number = 10

    json_name = 'connected_domin_score_dict.json'
    x = count_score_yolov4(MAX_TOTAL_AREA_RATE, MIN_SPLIT_AREA_RATE, selected_path, max_patch_number,json_name)
    print('total socre is', x)