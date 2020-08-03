"""
Training code for Adversarial patch training


"""

import PIL
import load_data
from tqdm import tqdm

from load_data import *
import gc
# import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
from tensorboardX import SummaryWriter
import subprocess
from utils.utils import *

import patch_config as patch_config
import sys
import time
import pickle

# from brambox.io.parser.annotation import DarknetParser as anno_darknet_parse
from utils import *
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
from skimage.segmentation import mark_boundaries
from fgsm import *
from train_patch_measure import measure_region_with_attack
import cv2
import sys

sys.setrecursionlimit(1000000)  # 例如这里设置为一百万


csv_name = 'x_result2.csv'
class PatchTrainer(object):
    def __init__(self, mode):
        self.config = patch_config.patch_configs[mode]()

        self.darknet_model = Darknet(self.config.cfgfile)
        self.darknet_model.load_weights(self.config.weightfile)
        self.darknet_model = self.darknet_model.eval().cuda() # TODO: Why eval?
        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer().cuda()
        self.prob_extractor = MaxProbExtractor(0, 80, self.config).cuda()
        self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.total_variation = TotalVariation().cuda()

        # self.writer = self.init_tensorboard(mode)

    def init_tensorboard(self, name=None):
        subprocess.Popen(['tensorboard', '--logdir=run_single'])
        if name is not None:
            time_str = time.strftime("%Y%m%d-%H%M%S")
            return SummaryWriter(f'run_single/{time_str}_{name}')
        else:
            return SummaryWriter()

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """

        img_size = self.darknet_model.height
        batch_size = self.config.batch_size
        n_epochs = 5000
        max_lab = 14

        time_str = time.strftime("%Y%m%d-%H%M%S")

        # Generate stating point
        adv_patch_cpu = self.generate_patch("gray")
        #adv_patch_cpu = self.read_image("saved_patches/patchnew0.jpg")

        adv_patch_cpu.requires_grad_(True)

        # zzj: position set
        patch_position_bias_cpu = torch.full((2, 1), 0)
        patch_position_bias_cpu[0]=0.0
        patch_position_bias_cpu[1]=0.01
        patch_position_bias_cpu.requires_grad_(True)



        # zzj: optimizer = optim.Adam([adv_patch_cpu, patch_position_bias], lr=self.config.start_learning_rate, amsgrad=True)

        optimizer = optim.Adam([
                                {'params': adv_patch_cpu, 'lr': self.config.start_learning_rate}
                               ], amsgrad=True)

        scheduler = self.config.scheduler_factory(optimizer)

        et0 = time.time()
        # import csv
        # with open(csv_name, 'w') as f:
        #     f_csv = csv.writer(f)
        #     f_csv.writerow([0,float(patch_position_bias_cpu[0]), float(patch_position_bias_cpu[1])])
        print(optimizer.param_groups[0]["lr"])

        ####### IMG ########

        img_dir = '/disk2/mycode/0511models/pytorch-YOLOv4-master-unofficial/select_from_test_500_0615'
        img_list = os.listdir(img_dir)
        img_list.sort()
        for img_name in img_list:
            print('------------------------')
            print('------------------------')
            print('Now testing', img_name)

            img_path = os.path.join(img_dir, img_name)

            img_batch = Image.open(img_path).convert('RGB')
            img_size = 608
            tf = transforms.Resize((img_size, img_size))
            img_batch_pil = tf(img_batch)
            tf = transforms.ToTensor()
            img_batch = tf(img_batch)



            import matplotlib.pyplot as plt

            image = img_as_float(io.imread(img_path))
            numSegments = 1000
            img_tensor_for_slic = img_batch.squeeze().permute(1, 2, 0)
            segments = slic(img_tensor_for_slic, n_segments=numSegments, sigma=3) + 1
            np_save_patch = os.path.join('slic_save', img_name.split('.')[0]+'_slic.npy')
            np.save(np_save_patch, segments)





            # fig = plt.figure("Superpixels -- %d segments" % (numSegments))
            # ax = fig.add_subplot(1, 1, 1)
            # ax.imshow(mark_boundaries(image, segments))
            # img = torch.from_numpy(mark_boundaries(image, segments)).permute(2, 0, 1).float()   # cpu [3,500,500]
            img = torch.from_numpy(segments).float()  # cpu [3,500,500]
            img = transforms.ToPILImage()(img.detach().cpu())
            img.save('seg.png')

            seg_result_num = np.max(segments)
            with torch.no_grad():
                boxes = do_detect(self.darknet_model, img_batch_pil, 0.4, 0.4, True)
            print('obj num begin:', len(boxes), float(boxes[0][4]), float(boxes[0][5]), float(boxes[0][6]))

            mask_detected = torch.Tensor(500, 500).fill_(0)
            black_img = torch.Tensor(3, 500, 500).fill_(0)
            white_img = torch.Tensor(3, 500, 500).fill_(1)
            white_img_single_layer = torch.Tensor(500, 500).fill_(1)
            black_img_single_layer = torch.Tensor(500, 500).fill_(0)

            noise_img = torch.Tensor(3, 500, 500).uniform_(0,1)
            gray_img = torch.Tensor(3, 500, 500).fill_(0.5)
            gray_img608 = torch.Tensor(3, 608, 608).fill_(0.5)
            original_box_list = []

            for box in boxes:
                bx_center = box[0] * 500
                by_center = box[1] * 500
                bw = box[2] * 500
                bh = box[3] * 500
                x1 = int(by_center-bh/2)
                x2 = int(by_center+bh/2)
                y1 = int(bx_center-bw/2)
                y2 = int(bx_center+bw/2)
                x1 = max(0, min(500, x1))
                x2 = max(0, min(500, x2))
                y1 = max(0, min(500, y1))
                y2 = max(0, min(500, y2))
                original_box_list.append([int(bx_center), int(by_center)])


                mask_detected[x1:x2,y1:y2]=1
            segments_tensor = torch.from_numpy(segments).float()

            old_boxes = boxes.copy()
            old_boxes_tensor = torch.Tensor(old_boxes)

            # img = mask_detected/torch.max(mask_detected)
            # img = transforms.ToPILImage()(img.detach().cpu())
            # img.show()

            segments_cover = torch.where((mask_detected == 1), segments_tensor, torch.FloatTensor(500, 500).fill_(0))

            # segments_cover = mask_detected.cpu()*torch.from_numpy(segments)
            segments_cover = segments_cover.numpy().astype(int)

            # img = torch.from_numpy(segments_cover).float()  # cpu [3,500,500]
            # img = img/torch.max(img)
            # img = transforms.ToPILImage()(img.detach().cpu())
            # img.show()

            unique_segments_cover = np.unique(segments_cover)
            unique_segments_cover = unique_segments_cover[1:]
            unique_segments_cover_list = []
            for reg_num_0 in unique_segments_cover:
                reg_img_0 = torch.where((segments_tensor == reg_num_0).mul(mask_detected == 1),
                                        white_img_single_layer,
                                        black_img_single_layer)
                reg_img_0_area = torch.sum(reg_img_0)
                if reg_img_0_area < 10:
                    continue
                unique_segments_cover_list.append(reg_num_0)
            unique_segments_cover = np.array(unique_segments_cover_list)





            # compute each super-pixel's attack ability

            resize_608 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((608, 608)),
                transforms.ToTensor()
            ])
            img_now = img_batch.clone()



            area_sum = 0
            patch_single_layer = torch.Tensor(500, 500).fill_(0)
            unique_segments_num = len(unique_segments_cover)
            print('list_len:', len(unique_segments_cover))

            # set a graph for super-pixel (region)
            from graph_test1 import Graph
            from itertools import combinations
            from skimage import measure
            c_2_n = list(combinations(unique_segments_cover, 2))
            graph_0 = Graph()

            for ver in unique_segments_cover:
                graph_0.addVertex(int(ver))

            # reg_img_134 = torch.where((segments_tensor == 134).mul(mask_detected == 1), white_img*0.1, black_img)
            # reg_img_139= torch.where((segments_tensor == 139).mul(mask_detected == 1), white_img*0.3, black_img)
            # reg_img_144= torch.where((segments_tensor == 144).mul(mask_detected == 1), white_img*0.5, black_img)
            # reg_img_150= torch.where((segments_tensor == 150).mul(mask_detected == 1), white_img*0.7, black_img)
            # reg_img_157= torch.where((segments_tensor == 157).mul(mask_detected == 1), white_img*0.8, black_img)
            # reg_img_151 = torch.where((segments_tensor == 151).mul(mask_detected == 1), white_img * 0.9, black_img)
            # reg_img_test = reg_img_134 + reg_img_139 + reg_img_144 + reg_img_150 + reg_img_157 + reg_img_151
            # img = reg_img_test  # cpu [500,500]
            # img = transforms.ToPILImage()(img.detach().cpu())
            # img.save('reg_img_test.png')
            # img.show()

            ## find the neighborhood
            # Dilation tensor
            dilation_tensor = torch.Tensor().cuda()
            conv2_kernel = torch.ones(1,1,3,3)
            cv2_kernel = np.ones((3,3), np.uint8)




            for reg_num_0 in tqdm(unique_segments_cover):
                reg_img_0 = torch.where((segments_tensor == reg_num_0).mul(mask_detected == 1), white_img_single_layer,
                                        black_img_single_layer)
                cv2_dilation_ = cv2.dilate(reg_img_0.numpy(), cv2_kernel, iterations=1)
                cv2_dilation_ = torch.from_numpy(cv2_dilation_).cuda()
                # img = reg_img_0  # cpu [500,500]
                # img = transforms.ToPILImage()(img.detach().cpu())
                # img.show()
                #
                # img = cv2_dilation_  # cpu [500,500]
                # img = transforms.ToPILImage()(img.detach().cpu())
                # img.show()
                dilation_tensor = torch.cat((dilation_tensor, cv2_dilation_.unsqueeze(0)))
            print()

            for reg_num_0 in tqdm(unique_segments_cover):
                reg_img_0 = torch.where((segments_tensor == reg_num_0).mul(mask_detected == 1),
                                        white_img_single_layer,
                                        black_img_single_layer).cuda()
                reg_img_0_batch = reg_img_0.repeat(dilation_tensor.shape[0],1,1)
                reg_img_0_batch_sum = reg_img_0_batch + dilation_tensor
                reg_img_0_batch_sum_max = torch.max(reg_img_0_batch_sum, dim=1).values
                reg_img_0_batch_sum_max = torch.max(reg_img_0_batch_sum_max, dim=1).values
                neibor1 = unique_segments_cover[reg_img_0_batch_sum_max.cpu() == 2]
                neibor1 = np.delete(neibor1, np.argwhere(neibor1==reg_num_0))
                for nei1 in neibor1:
                    graph_0.addEdge(reg_num_0, nei1, 1)
                    graph_0.addEdge(nei1, reg_num_0, 1)

                # for i_r in range(reg_img_0_batch_sum.shape[0]):
                #     if torch.max(reg_img_0_batch_sum[i_r]) == 2:
                #         reg_num_0 = reg_num_0
                #         reg_num_1 = unique_segments_cover[i_r]
                #         if reg_num_0 != reg_num_1:
                #             graph_0.addEdge(reg_num_0, reg_num_1, 1)
                #             graph_0.addEdge(reg_num_1, reg_num_0, 1)


            reg_num_0 = unique_segments_cover[0]
            # reg_num_1 = reg_combin[1]
            # reg_img_0 = torch.where((segments_tensor == reg_num_0).mul(mask_detected == 1), white_img, black_img)

            rw = graph_0

            graph_output_path = os.path.join('graph_save', img_name.split('.')[0]+'_graph.pkl')
            output_hal = open(graph_output_path, 'wb')

            str2 = pickle.dumps(rw)
            output_hal.write(str2)
            output_hal.close()



    def generate_patch(self, type):
        """
        Generate a random patch as a starting point for optimization.

        :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        :return:
        """
        if type == 'gray':
            adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)
        elif type == 'random':
            adv_patch_cpu = torch.rand((3, self.config.patch_size, self.config.patch_size))
        if type == 'trained_patch':
            patchfile = 'patches/object_score.png'
            patch_img = Image.open(patchfile).convert('RGB')
            patch_size = self.config.patch_size
            tf = transforms.Resize((patch_size, patch_size))
            patch_img = tf(patch_img)
            tf = transforms.ToTensor()
            adv_patch_cpu = tf(patch_img)

        return adv_patch_cpu

    def read_image(self, path):
        """
        Read an input image to be used as a patch

        :param path: Path to the image to be read.
        :return: Returns the transformed patch as a pytorch Tensor.
        """
        patch_img = Image.open(path).convert('RGB')
        tf = transforms.Resize((self.config.patch_size, self.config.patch_size))
        patch_img = tf(patch_img)
        tf = transforms.ToTensor()

        adv_patch_cpu = tf(patch_img)
        return adv_patch_cpu


def connected_domin_detect(input_img):
    from skimage import measure
    # detection
    if input_img.shape[0] ==3:
        input_img_new = (input_img[0]+input_img[1]+input_img[2])
    else:
        input_img_new = input_img
    ones = torch.Tensor(input_img_new.size()).fill_(1)
    zeros = torch.Tensor(input_img_new.size()).fill_(0)
    input_map_new = torch.where((input_img_new != 0), ones, zeros)
    # img = transforms.ToPILImage()(input_map_new.detach().cpu())
    # img.show()
    input_map_new = input_map_new.cpu()
    labels = measure.label(input_map_new[:, :], background=0, connectivity=2)
    label_max_number = np.max(labels)
    return float(label_max_number)


def get_obj_min_score(boxes):
    if type(boxes[0][0]) is list:
        min_score_list = []
        for i in range(len(boxes)):
            score_list = []
            for j in range(len(boxes[i])):
                score_list.append(boxes[i][j][4])
            min_score_list.append(min(score_list))
        return np.array(min_score_list)
    else:
        score_list = []
        for j in range(len(boxes)):
            score_list.append(boxes[j][4])
        return np.array(min(score_list))



def iou_all_tensor(old_boxes_tensor, new_boxes_tensor_all, threshold):
    confirm_confidence_max_list_lsit = []
    for new_boxes_tensor in new_boxes_tensor_all:
        new_boxes_tensor = new_boxes_tensor.permute(1, 0).contiguous()
        xxx = torch.argmin(torch.abs(new_boxes_tensor[0] - 0.27871165) + torch.abs(new_boxes_tensor[1] - 0.7041)
                           + torch.abs(new_boxes_tensor[2] - 0.3040) + torch.abs(new_boxes_tensor[3] - 0.4045))

        new_boxes_tensor = new_boxes_tensor.cpu()
        x_center = new_boxes_tensor[0]
        y_center = new_boxes_tensor[1]
        w = new_boxes_tensor[2]
        h = new_boxes_tensor[3]
        object_confidence = new_boxes_tensor[4]

        x0 = x_center - w/2
        y0 = y_center - h/2
        x1 = x_center + w / 2
        y1 = y_center + h / 2
        x0 = torch.clamp(x0, 0, 1)
        y0 = torch.clamp(y0, 0, 1)
        x1 = torch.clamp(x1, 0, 1)
        y1 = torch.clamp(y1, 0, 1)

        iou_max_index_list = []
        confirm_confidence_max_list = []

        total_size = new_boxes_tensor.shape[1]

        for i in range(old_boxes_tensor.shape[0]):
            old_tensor_expand = old_boxes_tensor[i].repeat(total_size, 1).transpose(0, 1).contiguous()
            x_center = old_tensor_expand[0]
            y_center = old_tensor_expand[1]
            w = old_tensor_expand[2]
            h = old_tensor_expand[3]
            x2 = x_center - w / 2
            y2 = y_center - h / 2
            x3 = x_center + w / 2
            y3 = y_center + h / 2
            x2 = torch.clamp(x2, 0, 1)
            y2 = torch.clamp(y2, 0, 1)
            x3 = torch.clamp(x3, 0, 1)
            y3 = torch.clamp(y3, 0, 1)
            # old_tensor_expand = old_boxes_tensor[i].repeat(total_size,1)


            # x0 = torch.Tensor([0.2])
            # x1 = torch.Tensor([0.4])
            # x2 = torch.Tensor([0.3])
            # x3 = torch.Tensor([0.7])
            #
            # y0 = torch.Tensor([0.2])
            # y1 = torch.Tensor([0.4])
            # y2 = torch.Tensor([0.3])
            # y3 = torch.Tensor([0.7])



            # computing area of each rectangles
            S_rec1 = (x1-x0) * (y1-y0)
            S_rec2 = (x3-x2) * (y3-y2)

            # computing the sum_area
            sum_area = S_rec1 + S_rec2

            # find the each edge of intersect rectangle
            left_line = torch.max(x0, x2)
            right_line = torch.min(x1, x3)
            top_line = torch.max(y0, y2)
            bottom_line = torch.min(y1, y3)

            # judge if there is an intersect
            intersect_flag = (top_line < bottom_line)*(left_line < right_line)
            intersect = (right_line - left_line) * (bottom_line - top_line)
            intersect = intersect * intersect_flag
            iou = intersect / (sum_area - intersect)

            iou_confirm = iou > threshold

            object_confidence_confirm = object_confidence * iou_confirm.byte()

            object_confidence_confirm_max_index = torch.argmax(object_confidence_confirm)

            # iou_max_index = torch.argmax(iou)
            # iou_max_index_list.append(iou_max_index)
            ## bigger than 0.4 confidence
            object_confidence_confirm_top, _ = torch.sort(object_confidence_confirm[object_confidence_confirm > 0.2], descending=True)
            object_confidence_confirm_top_3_or_half = object_confidence_confirm_top[:min(int(object_confidence_confirm_top.size(0)/2),3)]
            # confirm_confidence_max_list.append(torch.max(object_confidence_confirm))
            confirm_confidence_max_list.append(torch.mean(object_confidence_confirm_top_3_or_half))
        confirm_confidence_max_list_lsit.append(confirm_confidence_max_list)

    # left = torch.max(old_boxes_tensor[0][3].repeat(new_boxes_tensor.shape[1]), new_boxes_tensor[3])
    # for index in range(new_boxes_tensor.shape[1]):
    #     new_boxes_tensor_in = new_boxes_tensor[:, index]
    #     iou = compute_iou_tensor2(old_boxes_tensor[0], new_boxes_tensor_in[0:4])
    #     if iou > threshold:
    #         tm.append(iou)
    return confirm_confidence_max_list_lsit


def iou_all(old_boxes, new_boxes_all, threshold):
    length1 = len(new_boxes_all)
    confirm_list = []
    for i in range(len(new_boxes_all)):
        confirm_list.append([-1] * len(new_boxes_all[i]))

    for old_index in range(len(old_boxes)):
        old = old_boxes[old_index]
        for new_list_index in range(len(new_boxes_all)):
            new_list = new_boxes_all[new_list_index]
            for new_index in range(len(new_list)):
                new = new_list[new_index]
                iou_temp = iou_single(old, new)
                if iou_temp > threshold:
                    confirm_list[new_list_index][new_index] = old_index
    return confirm_list



def iou_single(boxa,boxb):
    # len = 7

    bx_center = boxa[0]
    by_center = boxa[1]
    bw = boxa[2]
    bh = boxa[3]
    x1 = by_center - bh / 2
    x2 = by_center + bh / 2
    y1 = bx_center - bw / 2
    y2 = bx_center + bw / 2
    xa1 = max(0, min(500, x1))
    xa2 = max(0, min(500, x2))
    ya1 = max(0, min(500, y1))
    ya2 = max(0, min(500, y2))

    bx_center = boxb[0]
    by_center = boxb[1]
    bw = boxb[2]
    bh = boxb[3]
    x1 = by_center - bh / 2
    x2 = by_center + bh / 2
    y1 = bx_center - bw / 2
    y2 = bx_center + bw / 2
    xb1 = max(0, min(500, x1))
    xb2 = max(0, min(500, x2))
    yb1 = max(0, min(500, y1))
    yb2 = max(0, min(500, y2))

    reg_a = [ya1, xa1, ya2, xa2]
    reg_b = [yb1, xb1, yb2, xb2]
    iou = compute_iou(reg_a, reg_b)
    return iou


def compute_iou_tensor2(reca, recb):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # transport
    rec1 = torch.Tensor([
                         max(0, min(1, reca[1] - reca[3] / 2.0)),
                         max(0, min(1, reca[0] - reca[2] / 2.0)),
                         max(0, min(1, reca[1] + reca[3] / 2.0)),
                         max(0, min(1, reca[0] + reca[2] / 2.0))
                         ])
    rec2 = torch.Tensor([
        max(0, min(1, recb[1] - recb[3] / 2.0)),
        max(0, min(1, recb[0] - recb[2] / 2.0)),
        max(0, min(1, recb[1] + recb[3] / 2.0)),
        max(0, min(1, recb[0] + recb[2] / 2.0))
    ])
    rec1 = rec1

    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0

def compute_iou_tensor(reca, recb):
    """
    computing IoU
    reca: (x_center, y_center, w, h)
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # transport
    # rec1 =

    # computing area of each rectangles
    # S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    # S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
    S_reca = reca[2] * reca[3]
    S_recb = recb[2] * recb[3]

    # computing the sum_area
    # sum_area = S_rec1 + S_rec2
    sum_area = S_reca + S_recb

    # find the each edge of intersect rectangle
    # left_line = max(rec1[1], rec2[1])
    # right_line = min(rec1[3], rec2[3])
    # top_line = max(rec1[0], rec2[0])
    # bottom_line = min(rec1[2], rec2[2])

    left_line = torch.max(reca[0]-reca[2]/2.0, recb[0]-recb[2]/2.0)
    right_line = torch.min(reca[0]+reca[2]/2.0, recb[0]+recb[2]/2.0)
    top_line = torch.max(reca[1]-reca[3]/2.0, recb[1]-recb[3]/2.0)
    bottom_line = torch.min(reca[1]+reca[3]/2.0, recb[1]+recb[3]/2.0)

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0

def main():
    if len(sys.argv) != 2:
        print('You need to supply (only) a configuration mode.')
        print('Possible modes are:')
        print(patch_config.patch_configs)


    trainer = PatchTrainer(sys.argv[1])
    trainer.train()

if __name__ == '__main__':
    main()


