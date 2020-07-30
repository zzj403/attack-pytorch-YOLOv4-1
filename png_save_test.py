from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import torch
import numpy as np
import os

selected_path = 'random_selected_img_800'
def main():
    patch_temp_size = 800

    files = os.listdir(selected_path)
    for img_name in files:

        img_path = os.path.join(selected_path, img_name)
        img0 = Image.open(img_path).convert('RGB')
    img1 = Image.open('person_and_bike_060_p.png').convert('RGB')
    resize2 = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Resize((patch_temp_size, patch_temp_size)),
        transforms.ToTensor()])
    img0_t = resize2(img0).cuda()
    img1_t = resize2(img1).cuda()
    img_minus_t = img0_t - img1_t
    num_classes = 80
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'

    img = transforms.ToPILImage()(img_minus_t.detach().cpu())
    img.show()
    x = connected_domin_detect(img_minus_t)
    print(x)


def connected_domin_detect(input_img):
    from skimage import measure
    # detection
    input_img_new = (input_img[0]+input_img[1]+input_img[2])
    ones = torch.cuda.FloatTensor(input_img_new.size()).fill_(1)
    zeros = torch.cuda.FloatTensor(input_img_new.size()).fill_(0)

    input_map_new = torch.where((input_img_new != 0), ones, zeros)
    img = transforms.ToPILImage()(input_map_new.detach().cpu())
    img.show()
    input_map_new = input_map_new.cpu()
    labels = measure.label(input_map_new[:, :], background=0, connectivity=2)
    print(labels)
    label_max_number = np.max(labels)
    return label_max_number





if __name__ == '__main__':
    main()