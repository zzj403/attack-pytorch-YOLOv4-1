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

def patch_aug(input_tensor):

    ## patch augementation
    min_contrast = 0.8
    max_contrast = 1.2
    min_brightness = -0.1
    max_brightness = 0.1
    noise_factor = 0.10
    batch_size = input_tensor.shape[0]
    adv_batch = input_tensor.unsqueeze(0)

    # Create random contrast tensor
    contrast = torch.Tensor(batch_size).uniform_(min_contrast, max_contrast)
    contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    contrast = contrast.expand(-1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))

    # Create random brightness tensor
    brightness = torch.Tensor(batch_size).uniform_(min_brightness, max_brightness)
    brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    brightness = brightness.expand(-1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))

    # Create random noise tensor
    noise = torch.Tensor(adv_batch.size()).uniform_(-1, 1) * noise_factor

    # Apply contrast/brightness/noise, clamp
    # adv_batch = adv_batch * contrast + brightness + noise
    adv_batch = adv_batch + brightness + noise

    # adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)
    adv_batch = adv_batch.squeeze()

    return adv_batch




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

        #adv_patch_cpu = self.read_image("saved_patches/patchnew0.jpg")



        # zzj: position set
        patch_position_bias_cpu = torch.full((2, 1), 0)
        patch_position_bias_cpu[0]=0.0
        patch_position_bias_cpu[1]=0.01
        patch_position_bias_cpu.requires_grad_(True)



        # zzj: optimizer = optim.Adam([adv_patch_cpu, patch_position_bias], lr=self.config.start_learning_rate, amsgrad=True)



        et0 = time.time()
        # import csv
        # with open(csv_name, 'w') as f:
        #     f_csv = csv.writer(f)
        #     f_csv.writerow([0,float(patch_position_bias_cpu[0]), float(patch_position_bias_cpu[1])])


        ####### IMG ########

        img_dir = '/disk2/mycode/0511models/pytorch-YOLOv4-master-unofficial/select_from_test_500_0615'
        super_pixel_dir = '/disk2/mycode/0511models/pytorch-YOLOv4-master-unofficial/black_superpixel'
        img_list = os.listdir(img_dir)
        img_list.sort()
        black_img = torch.Tensor(3, 608, 608).fill_(0)
        white_img = torch.Tensor(3, 608, 608).fill_(1)
        white_img_single_layer = torch.Tensor(500, 500).fill_(1)
        black_img_single_layer = torch.Tensor(500, 500).fill_(0)
        for img_name in img_list:
            print('------------------------')
            print('------------------------')
            print('Now training', img_name)


            ## read image and super-pixel
            img_path = os.path.join(img_dir, img_name)
            super_pixel_path = os.path.join(super_pixel_dir, img_name)

            img_batch_pil_500 = Image.open(img_path).convert('RGB')
            super_pixel_batch_pil_500 = Image.open(super_pixel_path)
            tf = transforms.Compose([
                transforms.Resize((500, 500)),
                transforms.ToTensor()
            ])
            img_batch = tf(img_batch_pil_500)
            super_pixel_batch = tf(super_pixel_batch_pil_500)
            super_pixel_batch_tmp = torch.where((super_pixel_batch > 0.01), white_img_single_layer, black_img_single_layer)
            if torch.sum(super_pixel_batch_tmp) > 5000:
                super_pixel_batch_tmp = torch.where((super_pixel_batch > 0.1), white_img_single_layer,
                                                    black_img_single_layer)
                if torch.sum(super_pixel_batch_tmp) > 5000:
                    super_pixel_batch_tmp = torch.where((super_pixel_batch > 0.5), white_img_single_layer,
                                                        black_img_single_layer)
            super_pixel_batch = super_pixel_batch_tmp

            ## generate patch
            adv_patch_cpu = self.generate_patch("gray")
            adv_patch_cpu.requires_grad_(True)



            # adv_patch_clip_cpu = torch.where((super_pixel_batch.repeat(3, 1, 1) == 1), adv_patch_cpu, black_img)

            # optimizer
            optimizer = optim.Adam([
                {'params': adv_patch_cpu, 'lr': self.config.start_learning_rate}
            ], amsgrad=True)

            scheduler = self.config.scheduler_factory(optimizer)


            ## img resize
            resize_500 = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((500, 500)),
                transforms.ToTensor()
            ])

            ## rotation start!
            for i in range(3000):

                ## augment
                adv_patch_cpu_batch = patch_aug(adv_patch_cpu.repeat(3, 1, 1, 1))
                ## patch apply
                img_batch_batch = img_batch.repeat(3, 1, 1, 1)
                noise = torch.Tensor(img_batch_batch.size()).uniform_(-1, 1) * 0.1
                img_batch_batch_noised = img_batch_batch + noise
                patched_img = torch.where((super_pixel_batch.repeat(3, 3, 1, 1) == 1), adv_patch_cpu_batch, img_batch_batch_noised)
                patched_img500 = patched_img
                patched_img608 = F.interpolate(patched_img500,
                                               (self.darknet_model.height, self.darknet_model.width),
                                               mode='bilinear').cuda()
                output = self.darknet_model(patched_img608)
                max_prob = self.prob_extractor(output)
                det_loss = torch.mean(max_prob)
                loss = det_loss  # + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())
                adv_patch_cpu_old = adv_patch_cpu.detach().clone()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if i % 20 == 0:
                    with torch.no_grad():
                        print(float(loss))
                        print("lr", optimizer.param_groups[0]["lr"])
                        img_test = torch.where((super_pixel_batch.repeat(3, 1, 1) == 1), adv_patch_cpu_old, img_batch)
                        img_test = img_test.unsqueeze(0)
                        img_inter_resize_t = F.interpolate(img_test,
                                                   (self.darknet_model.height, self.darknet_model.width),
                                                   mode='bilinear')
                        resize_small = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((608, 608)),
                            # transforms.ToTensor()
                        ])
                        resize_small_t = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((608, 608)),
                            transforms.ToTensor()
                        ])
                        img_pil_resize = resize_small(img_test.squeeze().cpu())
                        img_pil_resize_t = resize_small_t(img_test.squeeze().cpu())

                        img = (img_inter_resize_t - img_pil_resize_t)*1.0
                        img = img
                        imgshow = transforms.ToPILImage()(img.squeeze().detach().cpu())
                        imgshow.show()

                        boxes1 = do_detect(self.darknet_model, img_pil_resize, 0.5, 0.4, True)
                        print('box1 num:', len(boxes1))
                        class_names = load_class_names('data/coco.names')

                        output = self.darknet_model(img_inter_resize_t.cuda())
                        max_prob = self.prob_extractor(output)
                        print('max_prob', float(max_prob))
                        tf = transforms.ToPILImage()
                        # plot_boxes(transforms.ToPILImage()(img_test), boxes1, 'predictions.jpg', class_names)
                        print()

                '''
                ## patch apply

                patched_img = torch.where((super_pixel_batch.repeat(3, 1, 1) == 1), adv_patch_cpu, img_batch)
                patched_img500 = patched_img.unsqueeze(0).cuda()

                # img = torch.from_numpy(mark_boundaries(image, segments)).permute(2, 0, 1).float()  # cpu [3,500,500]
                # if i%20 ==0:
                #     img = transforms.ToPILImage()(patched_img.detach().cpu())
                #     img.show()
                patched_img608 = F.interpolate(patched_img500,
                                            (self.darknet_model.height, self.darknet_model.width),
                                            mode='bilinear')
                # img_test_t = F.interpolate(patched_img.unsqueeze(0),
                #                            (self.darknet_model.height, self.darknet_model.width),
                #                            mode='bilinear').cuda()


                adv_patch = adv_patch_cpu.cuda()

                ## train patch
                output = self.darknet_model(patched_img608)
                max_prob = self.prob_extractor(output)
                # nps = self.nps_calculator(adv_patch)
                # tv = self.total_variation(adv_patch)

                # nps_loss = nps * 0.01
                # tv_loss = tv * 2.5
                det_loss = torch.mean(max_prob)
                loss = det_loss #+ nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())
                # loss = det_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())

                # ep_det_loss += det_loss.detach().cpu().numpy()
                # ep_nps_loss += nps_loss.detach().cpu().numpy()
                # ep_tv_loss += tv_loss.detach().cpu().numpy()
                # ep_loss += loss
                adv_patch_cpu_old = adv_patch_cpu.detach().clone()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # adv_patch_cpu.data.clamp_(0, 1)  # keep patch in image range
                # scheduler.step(loss)

                if i%20 == 0:
                    print(float(max_prob))
                    print("lr", optimizer.param_groups[0]["lr"])
                    img_test = torch.where((super_pixel_batch.repeat(3, 1, 1) == 1), adv_patch_cpu_old, img_batch)
                    img_test = img_test.unsqueeze(0)
                    img_test_t = F.interpolate(img_test,
                                                (self.darknet_model.height, self.darknet_model.width),
                                                mode='bilinear')
                    resize_small = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((608, 608)),
                        # transforms.ToTensor()
                    ])
                    img_test = resize_small(img_test_t.squeeze().cpu())

                    # img = (img_test_t - img_test)*100
                    # img = img
                    # img = transforms.ToPILImage()(img.squeeze().detach().cpu())
                    # img.show()

                    boxes1 = do_detect(self.darknet_model, img_test, 0.5, 0.4, True)
                    print('box1 num:',len(boxes1))
                    class_names = load_class_names('data/coco.names')

                    output = self.darknet_model(img_test_t.cuda())
                    max_prob = self.prob_extractor(output)
                    print('max_prob', float(max_prob))
                    tf = transforms.ToPILImage()
                    # plot_boxes(transforms.ToPILImage()(img_test), boxes1, 'predictions.jpg', class_names)
                    print()'''

            ## Save Patched Image
            # tf = transforms.ToTensor()
            # img_batch_500 = tf(img_batch_pil_500)
            # super_pixel_batch_500 = tf(super_pixel_batch_pil_500)
            # adv_patch_cpu = resize_500(adv_patch_cpu)
            # patched_img = torch.where((super_pixel_batch_500.repeat(3, 1, 1) == 1), adv_patch_cpu, img_batch_500)

            img_save = torch.where((super_pixel_batch.repeat(3, 1, 1) == 1), adv_patch_cpu, img_batch)
            img = img_save.squeeze()  # cpu [3,500,500]
            img = transforms.ToPILImage()(img.detach().cpu())
            img.save(os.path.join('train_pixel_attack', img_name))

            img_path = os.path.join('train_pixel_attack', img_name)

            img_batch_pil_500 = Image.open(img_path).convert('RGB')
            resize_small2 = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.Resize((608, 608)),
                # transforms.ToTensor()
            ])
            img_batch_pil_608 = resize_small2(img_batch_pil_500)
            boxes0 = do_detect(self.darknet_model, img_batch_pil_608, 0.5, 0.4, True)
            print('box0 len:', len(boxes0))








    def generate_patch(self, type):
        """
        Generate a random patch as a starting point for optimization.

        :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        :return:
        """
        if type == 'gray':
            adv_patch_cpu = torch.full((3, 500, 500), 0.5)
        elif type == 'random':
            adv_patch_cpu = torch.rand((3, 500, 500))
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


