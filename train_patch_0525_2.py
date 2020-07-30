"""
Training code for Adversarial patch training


"""

import PIL
import load_data_0525_2
from tqdm import tqdm

from load_data_0525_2 import *
import gc
# import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
from tensorboardX import SummaryWriter
import subprocess

import patch_config
import sys
import time
from utils.utils import *

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
        subprocess.Popen(['tensorboard', '--logdir=runs'])
        if name is not None:
            time_str = time.strftime("%Y%m%d-%H%M%S")
            return SummaryWriter(f'runs/{time_str}_{name}')
        else:
            return SummaryWriter()

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """

        img_size = self.darknet_model.height
        batch_size = self.config.batch_size
        n_epochs = 10000
        max_lab = 14

        time_str = time.strftime("%Y%m%d-%H%M%S")

        # Generate stating point
        adv_patch_cpu = self.generate_patch("trained_patch")
        #adv_patch_cpu = self.read_image("saved_patches/patchnew0.jpg")

        adv_patch_cpu.requires_grad_(True)

        # train_loader = torch.utils.data.DataLoader(InriaDataset(self.config.img_dir, self.config.lab_dir, max_lab, img_size,
        #                                                         shuffle=True),
        #                                            batch_size=batch_size,
        #                                            shuffle=True,
        #                                            num_workers=10)
        # self.epoch_length = len(train_loader)
        # print(f'One epoch is {len(train_loader)}')

        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)
        selected_path = 'select1000_new'
        train_loader = os.listdir(selected_path)
        train_loader.sort()
        self.epoch_length = len(train_loader)
        img_size = 1000
        et0 = time.time()

        for i_batch, img_batch_name in tqdm(enumerate(train_loader), desc=f'Running epoch ',
                                                        total=self.epoch_length):
            # if i_batch > 120:
            #     break
            img_path = os.path.join(selected_path, img_batch_name)
            img_batch_pil = Image.open(img_path).convert('RGB')
            # img1 = Image.open('person_and_bike_060_p.png').convert('RGB')
            resize2 = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor()])
            img_batch_t = resize2(img_batch_pil).cuda()

            with torch.no_grad():
                img_batch = img_batch_t.cuda()
                img_batch = img_batch.unsqueeze(0)

                adv_patch = adv_patch_cpu.cuda()

                resize_small = transforms.Compose([
                    # transforms.ToPILImage(),
                    transforms.Resize((608, 608)),
                    transforms.ToTensor()])
                img_batch_detect = resize_small(img_batch_pil).cuda().unsqueeze(0)
                boxes = do_detect(self.darknet_model, img_batch_detect, 0.5, 0.4, use_cuda=True)
                if len(boxes) == 0:
                    img_batch_pil.save(os.path.join(selected_path + '_p', img_batch_name))
                    continue


                # namesfile = 'data/coco.names'
                # class_names = load_class_names(namesfile)
                # plot_boxes(img_batch_pil, boxes, 'predictions.jpg', class_names)

                boxes2 = []
                for box_index in range(len(boxes)):
                    box = boxes[box_index]
                    if box_index == 2:
                        break
                    boxes2.append([0, box[0],box[1],box[2],box[3]])
                lab_batch = torch.Tensor(boxes2).unsqueeze(0).cuda()

                # lab_batch = lab_batch[0,]

                # output = self.darknet_model(img_batch)
                # max_prob = self.prob_extractor(output)
                # lab_batch = max_prob


                adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, do_rotate=False, rand_loc=False)
                p_img_batch = self.patch_applier(img_batch, adv_batch_t)

                name = img_batch_name.split(sep='.')[0]
                properpatchedname = name + ".png"
                p_img_pil = transforms.ToPILImage('RGB')(p_img_batch.squeeze().cpu())
                p_img_pil.save(os.path.join(selected_path + '_p', properpatchedname))
                p_img_batch = F.interpolate(p_img_batch, (self.darknet_model.height, self.darknet_model.width))

                if i_batch < 5:

                    img = p_img_batch[0, :, :, ]
                    img = transforms.ToPILImage()(img.detach().cpu())
                    img.show()
                print()




    def generate_patch(self, type):
        """
        Generate a random patch as a starting point for optimization.

        :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        :return:
        """
        if type == 'trained_patch':
            patchfile = 'individualImage_v4_2.png'
            patch_img = Image.open(patchfile).convert('RGB')
            patch_size = self.config.patch_size
            tf = transforms.Resize((patch_size, patch_size))
            patch_img = tf(patch_img)
            tf = transforms.ToTensor()
            adv_patch_cpu = tf(patch_img)
        if type == 'black':
            adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 0.0)
        if type == 'gray':
            adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)
        elif type == 'random':
            adv_patch_cpu = torch.rand((3, self.config.patch_size, self.config.patch_size))

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


def main():
    if len(sys.argv) != 2:
        print('You need to supply (only) a configuration mode.')
        print('Possible modes are:')
        print(patch_config.patch_configs)


    trainer = PatchTrainer(sys.argv[1])
    trainer.train()

if __name__ == '__main__':
    main()


