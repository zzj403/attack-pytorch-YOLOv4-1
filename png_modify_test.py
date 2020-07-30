from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import torch
import numpy as np
import os

selected_path = 'random_selected_img_800'
def main():
    patch_temp_size = 800

    files = os.listdir(selected_path)
    files.sort()
    if not os.path.exists(selected_path + '_p'):
        os.makedirs(selected_path + '_p')

    for img_index in range(len(files)):
        print('img_index',img_index)
        if img_index > 50:
            break

        img_name = files[img_index]

        img_path = os.path.join(selected_path, img_name)
        img0 = Image.open(img_path).convert('RGB')
        # img1 = Image.open('person_and_bike_060_p.png').convert('RGB')
        resize2 = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.Resize((patch_temp_size, patch_temp_size)),
            transforms.ToTensor()])
        img0_t = resize2(img0).cuda()
        adv_patch_list = []
        patch_location_list = []
        for i in range(1):
            # patch_temp = torch.cuda.FloatTensor(3, (i+1)*60, (i+1)*60).fill_((i+1)/10+0.1)

            patch_temp_from_img = Image.open('pngtest1.png').convert('RGB')
            # patch_temp_size = 320,240
            resize3 = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.Resize((320, 240)),
                transforms.ToTensor()
            ])
            patch_temp = resize3(patch_temp_from_img).cuda()
            patch_temp_map = patch_temp[0] + patch_temp[1] + patch_temp[2]
            ones = torch.ones_like(patch_temp)
            zeros = torch.zeros_like(patch_temp)
            patch_temp = torch.where(patch_temp_map != 0, patch_temp, -ones)
            img = transforms.ToPILImage()(patch_temp.detach().cpu())
            # img.show()


            adv_patch_list.append(patch_temp)

            patch_location_list.append([0.05 * i, 0.20 * i])

        from patch_aply_0521 import PatchTransformer as PatchTransformer0521
        applyer0521 = PatchTransformer0521()
        p_img_batch = applyer0521(adv_patch_list, patch_location_list, img0_t.shape[1], img0_t)
        p_img = img0_t.squeeze(0)
        img = p_img_batch[0, :, :, ]
        img = transforms.ToPILImage()(img.detach().cpu())
        # img.show()
        name = img_name.split(sep='.')[0]
        properpatchedname = name + "_p.png"
        p_img_pil = transforms.ToPILImage('RGB')(p_img_batch.squeeze().cpu())
        p_img_pil.save(os.path.join(selected_path+'_p', properpatchedname))
        # img1_t = resize2(img1).cuda()

    # img_minus_t = img0_t - img1_t

    # img = transforms.ToPILImage()(img_minus_t.detach().cpu())
    # img.show()
    # x = connected_domin_detect(img_minus_t)
    # print(x)


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