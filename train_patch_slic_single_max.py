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
            if img_name != '1686.png' and img_name != '1687.png':
                continue
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
            numSegments = 400
            img_tensor_for_slic = img_batch.squeeze().permute(1, 2, 0)
            segments = slic(img_tensor_for_slic, n_segments=numSegments, sigma=3) + 1




            # fig = plt.figure("Superpixels -- %d segments" % (numSegments))
            # ax = fig.add_subplot(1, 1, 1)
            # ax.imshow(mark_boundaries(image, segments))
            img = torch.from_numpy(mark_boundaries(image, segments)).permute(2, 0, 1).float()   # cpu [3,500,500]
            img = transforms.ToPILImage()(img.detach().cpu())
            # img.show()

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

            # img = mask_detected/torch.max(mask_detected)
            # img = transforms.ToPILImage()(img.detach().cpu())
            # img.show()

            segments_cover = torch.where((mask_detected == 1), segments_tensor, torch.FloatTensor(500, 500).fill_(0))

            # segments_cover = mask_detected.cpu()*torch.from_numpy(segments)
            segments_cover = segments_cover.numpy().astype(int)

            img = torch.from_numpy(segments_cover).float()  # cpu [3,500,500]
            img = img/torch.max(img)
            img = transforms.ToPILImage()(img.detach().cpu())
            # img.show()

            unique_segments_cover = np.unique(segments_cover)
            unique_segments_cover = unique_segments_cover[1:]

            black_img = torch.Tensor(3, 500, 500).fill_(0)
            white_img = torch.Tensor(3, 500, 500).fill_(1)
            white_img_single_layer = torch.Tensor(500, 500).fill_(1)
            black_img_single_layer = torch.Tensor(500, 500).fill_(0)

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
                osp_img_gpu_batch = torch.Tensor().cuda()
                osp_area_list_tensor = torch.Tensor()
                batch_size_0 = 8
                max_prob_list_tensor = torch.Tensor()
                obj_min_score_list_tensor = torch.Tensor()

                # mei yi ge dan du gong ji
                for reg_num_index in range(len(unique_segments_cover)):
                    reg_num = unique_segments_cover[reg_num_index]
                    # one super-pixel image
                    osp_img = torch.where((segments_tensor.repeat(3, 1, 1) == reg_num).mul(mask_detected.repeat(3, 1, 1) == 1), black_img, img_now)

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
                        output_ = self.darknet_model(osp_img_gpu_batch)
                        max_prob = self.prob_extractor(output_)
                        max_prob_list_tensor = torch.cat((max_prob_list_tensor, max_prob.cpu()))

                        ## alternative method 2
                        boxes = do_detect(self.darknet_model, osp_img_gpu_batch, 0.4, 0.4, True)

                        obj_min_score = torch.from_numpy(get_obj_min_score(boxes)).float()

                        obj_min_score_list_tensor = torch.cat((obj_min_score_list_tensor, obj_min_score.cpu()))


                        osp_img_gpu_batch = torch.Tensor().cuda()

                ## 1
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

                _, max_index = torch.sort(max_prob_descend_div_osp_area_list_tensor, descending=True)
                connect_domin_num = 0
                area_sum = 0
                for ir in range(unique_segments_num):

                    area_sum += osp_area_list_tensor[max_index[ir]]
                    # area compute
                    max_region_num = unique_segments_cover[max_index[ir]]
                    osp_img_count_area = torch.where((segments_tensor == max_region_num).mul(mask_detected == 1),
                                                     white_img_single_layer,
                                                     black_img_single_layer)
                    patch_single_layer_2 = patch_single_layer + osp_img_count_area
                    area_sum_2 = torch.sum(patch_single_layer)
                    img = patch_single_layer_2  # cpu [3,500,500]
                    img = transforms.ToPILImage()(img.detach().cpu())
                    img.show()
                    img_now = torch.where((patch_single_layer_2== 1), black_img, img_batch)
                    img_now_gpu = resize_small(img_now).cuda().unsqueeze(0)
                    output_ = self.darknet_model(img_now_gpu)
                    max_prob = self.prob_extractor(output_)
                    print('max_prob=',max_prob)


                    connect_domin_num = connected_domin_detect(patch_single_layer_2)

                    if connect_domin_num > 10 or area_sum > 5000:
                        break
                    else:
                        patch_single_layer = patch_single_layer_2
                img_now = torch.where((patch_single_layer.repeat(3, 1, 1) == 1), black_img, img_batch)
                img = img_now  # cpu [3,500,500]
                img = transforms.ToPILImage()(img.detach().cpu())
                img.show()



                min_index = torch.argmax(obj_min_score_descend_div_osp_area_list_tensor)
                min_region_num = unique_segments_cover[int(min_index)]
                # area compute
                osp_img_count_area = torch.where((segments_tensor == min_region_num).mul(mask_detected==1), white_img_single_layer,
                                                 black_img_single_layer)
                patch_single_layer = patch_single_layer + osp_img_count_area
                area_sum = float(torch.sum(patch_single_layer))
                print('area_sum=', area_sum)

                connect_domin_num = connected_domin_detect(patch_single_layer)
                if connect_domin_num > 10:
                    break
                if area_sum > 5000:
                    break

                unique_segments_cover = np.delete(unique_segments_cover, int(min_index))
                img_now = torch.where((segments_tensor.repeat(3, 1, 1) == min_region_num)
                                      .mul(mask_detected.repeat(3, 1, 1) == 1), black_img, img_now)
                img = img_now  # cpu [3,500,500]
                img = transforms.ToPILImage()(img.detach().cpu())
                img.show()

                img_now_gpu = resize_small(img_now).cuda().unsqueeze(0)

                # alternative method 1
                # output_ = self.darknet_model(img_now_gpu)
                # max_prob = self.prob_extractor(output_)

                # alternative method 2
                boxes = do_detect(self.darknet_model, img_now_gpu, 0.4, 0.4, True)
                obj_min_score_now = torch.from_numpy(get_obj_min_score(boxes)).float()
                obj_min_score_now = float(obj_min_score_now.squeeze())


                print('min_visiable_score now:', obj_min_score_now)
                tf = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((608, 608)),
                    transforms.ToTensor()
                ])
                img_now_608 = tf(img_now.detach().cpu()).unsqueeze(0)

                boxes = do_detect(self.darknet_model, img_now_608, 0.4, 0.4, True)
                class_names = load_class_names('data/coco.names')
                plot_boxes(img, boxes, 'predictions.jpg', class_names)
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
        lab_batch[0, 0, 4] = 0.76

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






def main():
    if len(sys.argv) != 2:
        print('You need to supply (only) a configuration mode.')
        print('Possible modes are:')
        print(patch_config.patch_configs)


    trainer = PatchTrainer(sys.argv[1])
    trainer.train()

if __name__ == '__main__':
    main()


