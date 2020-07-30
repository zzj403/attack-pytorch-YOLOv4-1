import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
txt = 'data.txt'
with open(txt) as f:
    list1 = f.read().splitlines()

file_path = '/disk2/mycode/0511models/pytorch-YOLOv4-master-unofficial/select2000_new'

save_path = 'select1000_500'

for item in tqdm(list1):

    img_path0 = os.path.join(file_path, item)
    img0 = Image.open(img_path0).convert('RGB')
    resize_small = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((500, 500)),
        # transforms.ToTensor()
    ])
    img1 = resize_small(img0)
    img1.save(os.path.join(save_path, item))