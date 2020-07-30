import torch
import numpy as np

#segments = get_grid(img_tensor_for_slic, width=10) + 1
def get_grid(img_tensor, width=10):
    if len(img_tensor.shape) == 3:
        w = img_tensor.shape[0]
        h = img_tensor.shape[1]
    elif len(img_tensor.shape) == 2:
        w = img_tensor.shape[0]
        h = img_tensor.shape[1]
    grid_map = torch.Tensor(w,h).fill_(0)
    no = 0
    for i in range(int(w/width)):
        for j in range(int(h/width)):
            grid_map[i*width:(i+1)*width, j*width:(j+1)*width] = no
            no = no + 1
    return grid_map.numpy().astype(int)


def get_grid_mini(img_tensor, width=10):
    if len(img_tensor.shape) == 3:
        w = img_tensor.shape[0]
        h = img_tensor.shape[1]
    elif len(img_tensor.shape) == 2:
        w = img_tensor.shape[0]
        h = img_tensor.shape[1]
    grid_map = torch.Tensor(int(w/width),int(h/width)).fill_(0)
    no = 0
    for i in range(int(w/width)):
        for j in range(int(h/width)):
            grid_map[i:(i+1), j:(j+1)] = no
            no = no + 1
    return grid_map.numpy().astype(int)


def get_grid_neighbor(segments_mini, selected_node, unique_segments_cover):
    # selected_node = 1
    x,y = np.where(segments_mini == selected_node)
    neighbor_list = []
    for i in range(3):
        for j in range(3):
            x_now = x - 1 + i
            y_now = y - 1 + j
            if i!=1 and j!=1:
                continue
            if 0 <= x_now < segments_mini.shape[0]:
                if 0 <= y_now < segments_mini.shape[1]:
                    if int(segments_mini[x_now,y_now]) in unique_segments_cover:
                        neighbor_list.append(int(segments_mini[x_now,y_now]))
    neighbor_list.remove(selected_node)
    return neighbor_list
