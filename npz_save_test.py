import json
from PIL import Image, ImageDraw, ImageFont
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
from utils.utils import *
from tool.darknet2pytorch import *
from load_data_0517 import PatchTransformer, PatchApplier, InriaDataset
import json

adv_patch_list = []
patch_location_list = []
for i in range(6):
    # patch_temp = torch.cuda.FloatTensor(3, (i+1)*60, (i+1)*60).fill_((i+1)/10+0.1)
    patch_temp = torch.cuda.FloatTensor([[0, 0, 1, 1, 0, 0, 0, 0],
                                         [0, 0.3, 0.5, 0.7, 0, 0, 0, 0],
                                         [1, 1, 1, 1, 0, 0, 0, 0],
                                         [1, 1, 1, 1, 1, 1, 1, 1],
                                         [0, 1, 1, 1, 1, 1, 1, 0],
                                         [0, 0, 1, 1, 1, 1, 0, 0],
                                         [0, 0, 0, 1, 1, 0, 0, 0]])  # *((i+1)/10+0.1)
    patch_temp = patch_temp.unsqueeze(0)
    patch_temp = torch.cat((patch_temp, patch_temp, patch_temp), 0).cpu()

    patch_temp_from_img = Image.open('test2.bmp').convert('RGB')
    patch_temp_size = (i + 1) * 60
    resize2 = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((patch_temp_size, patch_temp_size)),
        transforms.ToTensor()])
    patch_temp = resize2(patch_temp_from_img).cuda()
    patch_temp_map = patch_temp[0] + patch_temp[1] + patch_temp[2]
    ones = torch.ones_like(patch_temp)
    zeros = torch.zeros_like(patch_temp)
    patch_temp = torch.where(patch_temp_map != 0, patch_temp, -ones)
    adv_patch_list.append(patch_temp.cpu())
    # patch_location_list.append([i/16, i/8])
    patch_location_list.append([0.13 * i, 0.01 * i])

    np.save('test1_' + str(i) + '.npy', patch_temp.cpu())
    np.save('test1_' + str(i) + 'loc.npy', [0.13 * i, 0.01 * i])
    a = np.load('test1_' + str(i) + '.npy')
    b = np.load('test1_' + str(i) + 'loc.npy')
    a = torch.from_numpy(a).cuda()
    b = torch.from_numpy(b).cuda()

# a=np.arange(5)
# np.save('test1_'+str(i)+'.npy',patch_location_list)
#
# np.savez('array_save.npz', adv_patch_list[0], adv_patch_list[1])
# A = np.load('array_save.npz')
print()
# with open('number.json', 'w') as A_obj:
#     json.dump(adv_patch_list, A_obj)
#     json.dump(patch_location_list, A_obj)
#     print(A_obj)
# print()


