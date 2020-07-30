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




            # fig = plt.figure("Superpixels -- %d segments" % (numSegments))
            # ax = fig.add_subplot(1, 1, 1)
            # ax.imshow(mark_boundaries(image, segments))
            # img = torch.from_numpy(mark_boundaries(image, segments)).permute(2, 0, 1).float()   # cpu [3,500,500]
            img = torch.from_numpy(segments).float()  # cpu [3,500,500]
            img = transforms.ToPILImage()(img.detach().cpu())
            img.save('seg.png')

            seg_result_num = np.max(segments)

            boxes = do_detect(self.darknet_model, img_batch_pil, 0.4, 0.4, True)
            print('obj num begin:', len(boxes), float(boxes[0][4]), float(boxes[0][5]), float(boxes[0][6]))

            mask_detected = torch.Tensor(500, 500).fill_(0)

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

            black_img = torch.Tensor(3, 500, 500).fill_(0)
            white_img = torch.Tensor(3, 500, 500).fill_(1)
            white_img_single_layer = torch.Tensor(500, 500).fill_(1)
            black_img_single_layer = torch.Tensor(500, 500).fill_(0)

            noise_img = torch.Tensor(3, 500, 500).uniform_(0,1)

            # compute each super-pixel's attack ability

            resize_small = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((608, 608)),
                transforms.ToTensor()
            ])
            img_now = img_batch.clone()


            with torch.no_grad():
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

                # find the neighborhood
                '''
                for reg_combin in tqdm(c_2_n):
                    reg_num_0 = reg_combin[0]
                    reg_num_1 = reg_combin[1]
                    reg_img_0 = torch.where((segments_tensor == reg_num_0).mul(mask_detected == 1), white_img, black_img)
                    reg_img_1 = torch.where((segments_tensor == reg_num_1).mul(mask_detected == 1), white_img, black_img)
                    reg_img_sum = reg_img_0 + reg_img_1
                    labels_0 = measure.label(reg_img_0, background=0, connectivity=2)
                    labels_1 = measure.label(reg_img_1, background=0, connectivity=2)
                    labels_sum = measure.label(reg_img_sum, background=0, connectivity=2)
                    label_max_number_0 = np.max(labels_0)
                    label_max_number_1 = np.max(labels_1)
                    label_max_number_sum = np.max(labels_sum)
                    if label_max_number_sum < label_max_number_0 + label_max_number_1:
                        graph_0.addEdge(reg_num_0, reg_num_1, 1)
                        graph_0.addEdge(reg_num_1, reg_num_0, 1)

                rw = graph_0
                output_hal = open("graph.pkl", 'wb')
                
                str = pickle.dumps(rw)
                output_hal.write(str)
                output_hal.close()'''
                with open("graph.pkl", 'rb') as file:
                    rq = pickle.loads(file.read())
                graph_0 = rq

                '''
                # print('list_len:', len(unique_segments_cover))
                osp_img_gpu_batch = torch.Tensor().cuda()
                osp_area_list_tensor = torch.Tensor()
                batch_size_0 = 8
                max_prob_list_tensor = torch.Tensor()
                obj_min_score_list_tensor = torch.Tensor()
                do_boxes_list = []
                output_all = torch.Tensor()

                for reg_num_index in range(len(unique_segments_cover)):
                    reg_num = unique_segments_cover[reg_num_index]
                    # one super-pixel image
                    osp_img = torch.where((segments_tensor.repeat(3, 1, 1) == reg_num).mul(mask_detected.repeat(3, 1, 1) == 1), noise_img, img_now)

                    # show the img
                    # if reg_num_index == 2:
                    #     osp_img_gpu = resize_small(osp_img).cuda().unsqueeze(0)
                    #     boxes = do_detect(self.darknet_model, osp_img_gpu, 0.4, 0.4, True)
                    #     class_names = load_class_names('data/coco.names')
                    #     plot_boxes(transforms.ToPILImage()(osp_img), boxes, 'predictions0.jpg', class_names)
                    #     print()

                    # compute the area
                    osp_layer = torch.where(
                        (segments_tensor == reg_num).mul(mask_detected == 1),
                        white_img_single_layer, black_img_single_layer)
                    osp_area = torch.sum(osp_layer).unsqueeze(0)
                    osp_area_list_tensor = torch.cat((osp_area_list_tensor, osp_area))


                    # img = osp_img_count_area  # cpu [3,500,500]
                    # img = transforms.ToPILImage()(img.detach().cpu())
                    # img.show()


                    osp_img_gpu = resize_small(osp_img).cuda().unsqueeze(0)
                    osp_img_gpu_batch = torch.cat((osp_img_gpu_batch, osp_img_gpu), dim=0)
                    if osp_img_gpu_batch.shape[0] >= batch_size_0 or reg_num_index+1 == len(unique_segments_cover):
                        ## alternative method 1
                        # output_ = self.darknet_model(osp_img_gpu_batch)img_batch
                        old_boxes222 = do_detect(self.darknet_model, osp_img_gpu_batch,0.4,0.4,True)
                        output_ = do_detect_all(self.darknet_model, osp_img_gpu_batch,0.0,0.1,True)
                        output_l = [torch.tensor(output_[0]),torch.tensor(output_[1]),torch.tensor(output_[2])]
                        # output_2 = self.darknet_model(osp_img_gpu_batch)
                        ## get yolo output batch
                        YOLOoutput = output_l
                        num_anchors = 3  # zzj! you should change here
                        num_classes = 80  # zzj! you should change here
                        output_batch = torch.Tensor()
                        output_batch = output_batch
                        for i in range(len(YOLOoutput)):
                            YOLOoutput_t = YOLOoutput[i]
                            if YOLOoutput_t.dim() == 3:
                                YOLOoutput_t = YOLOoutput_t.float().cpu()
                            batch = YOLOoutput_t.size(0)

                            output = YOLOoutput_t



                            # output = output.transpose(1, 2).contiguous()  # [batch, 85, 3, 361] -- [7, batch, 17328]
                            # output = output.reshape(7, -1)  # [batch, 85, 3*361] -- [batch
                            output_batch = torch.cat([output_batch, output], dim=1)
                        output_all = torch.cat([output_all, output_batch], dim=0)



                        # boxes = do_detect(self.darknet_model, osp_img_gpu_batch, 0.4, 0.4, True)
                        # do_boxes_list = do_boxes_list + boxes

                        # max_prob = self.prob_extractor(output_)
                        # max_prob_list_tensor = torch.cat((max_prob_list_tensor, max_prob.cpu()))

                        ## alternative method 2
                        # boxes = do_detect(self.darknet_model, osp_img_gpu_batch, 0.4, 0.4, True)
                        # obj_min_score = torch.from_numpy(get_obj_min_score(boxes)).float()
                        # obj_min_score_list_tensor = torch.cat((obj_min_score_list_tensor, obj_min_score.cpu()))


                        osp_img_gpu_batch = torch.Tensor().cuda()


                ## no nms
                # output_all = output_all  # [46, 22743, 7]
                # output_all = output_all.permute(2, 0, 1).contiguous().reshape(7, -1)[:5]
                confirm_confidence_list = iou_all_tensor(old_boxes_tensor, output_all, threshold=0.8)
                confirm_confidence_tensor = torch.tensor(confirm_confidence_list)
                old_boxes_confidence_list = [torch.tensor(x[4]) for x in old_boxes]
                old_boxes_confidence_tensor = torch.tensor(old_boxes_confidence_list)
                old_boxes_confidence_tensor_extend = old_boxes_confidence_tensor.repeat(confirm_confidence_tensor.shape[0],1)
                confidence_reduce_tensor = old_boxes_confidence_tensor_extend - confirm_confidence_tensor
                # reduce_max_index = torch.argmax(confidence_reduce_tensor, dim=0)
                reduce_max, reduce_max_index = torch.max(confidence_reduce_tensor, dim=0)
                select_superpixel = unique_segments_cover[reduce_max_index]'''




                ## apply init select sp
                sp_layer_now = torch.Tensor(500, 500).fill_(0)

                img_now = img_batch.clone()

                # select_list = list(select_superpixel)
                select_list = [586, 661, 661]
                for selected_node in select_list:
                    sp_layer_now = torch.where((segments_tensor == selected_node).mul(mask_detected == 1), white_img_single_layer,
                        sp_layer_now)
                # compute the area
                sp_layer_area = torch.sum(sp_layer_now)


                ## iteration search


                batch_size_0 = 8
                do_boxes_list = []



                for step in range(unique_segments_num):

                    ## get all neighborhood
                    now_neighbor_list = []
                    output_all = torch.Tensor()
                    osp_area_list_tensor = torch.Tensor()
                    osp_img_gpu_batch = torch.Tensor().cuda()
                    for selected_node in select_list:
                        neighbor_list = list(graph_0.getVertex(selected_node).connectedTo.keys())
                        for nei in neighbor_list:
                            now_neighbor_list.append(nei.id)

                    now_neighbor_np = np.array(now_neighbor_list)
                    now_neighbor_unique_np = np.unique(now_neighbor_np)
                    now_neighbor_unique_list = list(now_neighbor_unique_np)
                    for ite1 in select_list:
                        if now_neighbor_unique_list.count(ite1) > 0:
                            now_neighbor_unique_list.remove(ite1)
                    now_neighbor_unique_np = np.array(now_neighbor_unique_list)

                    img_tmp = torch.where(
                            (sp_layer_now.repeat(3, 1, 1) == 1).mul(mask_detected.repeat(3, 1, 1) == 1),
                            noise_img, img_batch.clone())
                    boxes222 = do_detect(self.darknet_model, resize_small(img_tmp).cuda(), 0.4, 0.4, True)
                    class_names = load_class_names('data/coco.names')
                    plot_boxes(transforms.ToPILImage()(img_tmp.detach().cpu()), boxes222, 'test2/predictions'+str(step)+'.jpg',
                               class_names)
                    print()

                    ## iteration with all neighborhood
                    noise_repeat_times = 7
                    for i1 in range(now_neighbor_unique_np.size):
                        for nz_t in range(noise_repeat_times):
                            now_node = now_neighbor_unique_np[i1]
                            noise_img = torch.Tensor(3, 500, 500).uniform_(0,1)
                            sp_layer_tmp = torch.where((segments_tensor == now_node).mul(mask_detected == 1), white_img_single_layer,
                                sp_layer_now)
                            osp_img = torch.where(
                                (sp_layer_tmp.repeat(3, 1, 1) == 1).mul(mask_detected.repeat(3, 1, 1) == 1),
                                noise_img, img_batch.clone())
                            osp_img_gpu = resize_small(osp_img).cuda().unsqueeze(0)
                            osp_img_gpu_batch = torch.cat((osp_img_gpu_batch, osp_img_gpu), dim=0)

                        # compute the area
                        osp_layer = torch.where(
                            (segments_tensor == now_node).mul(mask_detected == 1),
                            white_img_single_layer, black_img_single_layer)
                        osp_area = torch.sum(osp_layer).unsqueeze(0)
                        osp_area_list_tensor = torch.cat((osp_area_list_tensor, osp_area))



                        output_ = do_detect_all(self.darknet_model, osp_img_gpu_batch, 0.0, 0.1, True)
                        output_l = [torch.tensor(output_[0]), torch.tensor(output_[1]), torch.tensor(output_[2])]

                        ## get yolo output batch
                        YOLOoutput = output_l
                        num_anchors = 3  # zzj! you should change here
                        num_classes = 80  # zzj! you should change here
                        output_batch = torch.Tensor()
                        output_batch = output_batch
                        for i in range(len(YOLOoutput)):
                            YOLOoutput_t = YOLOoutput[i]
                            if YOLOoutput_t.dim() == 3:
                                YOLOoutput_t = YOLOoutput_t.float().cpu()
                            output = YOLOoutput_t
                            if YOLOoutput_t.dim() != 3:
                                pass
                            output_batch = torch.cat([output_batch, output], dim=1)
                        output_all = torch.cat([output_all, output_batch], dim=0)

                        osp_img_gpu_batch = torch.Tensor().cuda()
                    print()
                    ## no nms
                    # output_all = output_all  # [46, 22743, 7]
                    # output_all = output_all.permute(2, 0, 1).contiguous().reshape(7, -1)[:5]
                    confirm_confidence_list = iou_all_tensor(old_boxes_tensor, output_all, threshold=0.4)
                    confirm_confidence_tensor = torch.tensor(confirm_confidence_list)

                    ## mean of dif noise
                    confirm_confidence_tensor_new = torch.Tensor()
                    for i3 in range(int(confirm_confidence_tensor.size(0)/noise_repeat_times)):
                        num_t = i3*noise_repeat_times
                        confirm_confidence_tensor_tmp = torch.mean(confirm_confidence_tensor[num_t:num_t+noise_repeat_times], dim=0)
                        confirm_confidence_tensor_new = torch.cat([confirm_confidence_tensor_new,
                                                                   confirm_confidence_tensor_tmp.unsqueeze(0)],dim=0)
                    confirm_confidence_tensor = confirm_confidence_tensor_new
                    old_boxes_confidence_list = [torch.tensor(x[4]) for x in old_boxes]
                    old_boxes_confidence_tensor = torch.tensor(old_boxes_confidence_list)
                    old_boxes_confidence_tensor_extend = old_boxes_confidence_tensor.repeat(
                        confirm_confidence_tensor.shape[0], 1)
                    confidence_reduce_tensor = old_boxes_confidence_tensor_extend - confirm_confidence_tensor

                    ## delete too low confidence anchor
                    zeros_tensor = torch.Tensor(confirm_confidence_tensor.size()).fill_(0)
                    confidence_reduce_tensor = torch.where(confirm_confidence_tensor < 0.3, zeros_tensor, confidence_reduce_tensor)

                    ## div area


                    confidence_reduce_div_area_tensor = confidence_reduce_tensor / osp_area_list_tensor.repeat(3,1).permute(1, 0)

                    reduce_max, reduce_max_index = torch.max(confidence_reduce_tensor, dim=0)
                    # reduce_max, reduce_max_index = torch.max(confidence_reduce_div_area_tensor, dim=0)

                    ## find max of 1
                    # select_superpixel_index = torch.argmax(confidence_reduce_tensor) / confidence_reduce_tensor.shape[1]

                    ## find max of average 3
                    select_superpixel_index = torch.argmax(torch.sum(confidence_reduce_tensor, dim=1))
                    select_superpixel_no = now_neighbor_unique_np[select_superpixel_index]
                    select_list.append(select_superpixel_no)

                    sp_layer_now = torch.where((segments_tensor == select_superpixel_no).mul(mask_detected == 1),
                                               white_img_single_layer,
                                               sp_layer_now)

                    print()















                    '''
                    ## 1
                    # confirm_list = iou_all(old_boxes, do_boxes_list, threshold=0.4)

                    old_boxes_length = len(old_boxes)
                    min_match_score_list = [1] * old_boxes_length
                    min_match_index_list = [1] * old_boxes_length
                    for confirm_list_index in range(len(confirm_list)):
                        confirm_list_in = confirm_list[confirm_list_index]
                        for confirm_list_in_index in range(len(confirm_list_in)):
                            confirm_i = confirm_list[confirm_list_index][confirm_list_in_index]
                            if confirm_i == -1:
                                continue
                            confirm_socre = do_boxes_list[confirm_list_index][confirm_i][4]

                            if confirm_socre < min_match_score_list[confirm_i]:
                                min_match_score_list[confirm_i] = confirm_socre
                                min_match_index_list[confirm_i] = [confirm_list_index, confirm_list_in_index]

                    output_before = self.darknet_model(resize_small(img_now).unsqueeze(0).cuda())
                    max_prob_before = self.prob_extractor(output_before).cpu()
                    max_prob_descend_list_tensor = max_prob_before - max_prob_list_tensor
                    max_prob_descend_div_osp_area_list_tensor = max_prob_descend_list_tensor / osp_area_list_tensor

                    ## 2
                    # boxes = do_detect(self.darknet_model, resize_small(img_now).unsqueeze(0).cuda(), 0.4, 0.4, True)
                    # obj_min_score_before = torch.from_numpy(get_obj_min_score(boxes)).float()
                    # obj_min_score_before = obj_min_score_before.squeeze()
                    # obj_min_score_descend_list_tensor = obj_min_score_before - obj_min_score_list_tensor
                    # obj_min_score_descend_div_osp_area_list_tensor = obj_min_score_descend_list_tensor/osp_area_list_tensor


                    # max_index = torch.argmax(max_prob_descend_div_osp_area_list_tensor)
                    _, max_index_list_tensor = torch.sort(max_prob_descend_div_osp_area_list_tensor, descending=True)
                    for ir in range(len(max_index_list_tensor)):
                        if osp_area_list_tensor[max_index_list_tensor[ir]] < 200:
                            continue
                        else:
                            max_index = max_index_list_tensor[ir]
                            break

                    min_region_num = unique_segments_cover[int(max_index)]
                    # area compute
                    osp_img_count_area = torch.where((segments_tensor == min_region_num).mul(mask_detected==1), white_img_single_layer,
                                                     black_img_single_layer)
                    patch_single_layer_2 = patch_single_layer + osp_img_count_area
                    area_sum = float(torch.sum(patch_single_layer_2))
                    print('area_sum=', area_sum)

                    connect_domin_num = connected_domin_detect(patch_single_layer)
                    if connect_domin_num > 10:
                        break
                    if area_sum > 5000:
                        break
                    patch_single_layer = patch_single_layer_2

                    unique_segments_cover = np.delete(unique_segments_cover, int(max_index))

                    img_now = torch.where((segments_tensor.repeat(3, 1, 1) == min_region_num)
                                          .mul(mask_detected.repeat(3, 1, 1) == 1), black_img, img_now)
                    # img = img_now  # cpu [3,500,500]
                    # img = transforms.ToPILImage()(img.detach().cpu())
                    # img.show()

                    img_now_gpu = resize_small(img_now).cuda().unsqueeze(0)

                    ## alternative method 1
                    output_ = self.darknet_model(img_now_gpu)
                    max_prob = self.prob_extractor(output_)

                    ## alternative method 2
                    # boxes = do_detect(self.darknet_model, img_now_gpu, 0.4, 0.4, True)
                    # obj_min_score_now = torch.from_numpy(get_obj_min_score(boxes)).float()
                    # obj_min_score_now = float(obj_min_score_now.squeeze())


                    print('max_visiable_score now:', float(max_prob))
                    tf = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((608, 608)),
                        transforms.ToTensor()
                    ])
                    img_now_608 = tf(img_now.detach().cpu()).unsqueeze(0)

                    boxes = do_detect(self.darknet_model, img_now_608, 0.5, 0.4, True)
                    class_names = load_class_names('data/coco.names')
                    # plot_boxes(img, boxes, 'predictions.jpg', class_names)
                    if len(boxes) != 0:
                        print('-------------')
                        print('obj num now:', len(boxes))
                        for x in range(len(boxes)):
                            print(float(boxes[x][4]), float(boxes[x][5]), class_names[int(boxes[x][6])])
                        print('connect_domin_num = ', connect_domin_num)
                    else:
                        break
            img = img_now  # cpu [3,500,500]
            img = transforms.ToPILImage()(img.detach().cpu())
            img.save(os.path.join('black_superpixel_img', img_name))

            img = patch_single_layer
            img = transforms.ToPILImage()(img.detach().cpu())
            img.save(os.path.join('black_superpixel', img_name))

            reg_cover = torch.Tensor(500, 500).fill_(0)
            for reg_num in unique_segments_cover:
                reg_cover_temp = torch.where((segments_tensor == reg_num), segments_tensor, torch.FloatTensor(500,500).fill_(0))
                reg_cover += reg_cover_temp

        lab_batch = torch.cuda.FloatTensor(1, 14, 5).fill_(1)
        lab_batch[0, 0, 0] = 0
        lab_batch[0, 0, 1] = 0.25
        lab_batch[0, 0, 2] = 0.4
        lab_batch[0, 0, 3] = 0.43
        lab_batch[0, 0, 4] = 0.76'''

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
            object_confidence_confirm_top_half = object_confidence_confirm_top[:int(object_confidence_confirm_top.size(0)/2)]
            # confirm_confidence_max_list.append(torch.max(object_confidence_confirm))
            confirm_confidence_max_list.append(torch.mean(object_confidence_confirm_top_half))
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


