import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class SingleStyleDataset(Dataset):
    def __init__(self, opts, cfg):
        self.opts = opts
        self.cfg = cfg
        self.value_scale = 255
        self.style_list = sorted(os.listdir(opts.style_img_folder))
        self.style_list = [os.path.join(opts.style_img_folder, i) for i in self.style_list]
        self.content_list = sorted(os.listdir(opts.content_img_folder))
        self.content_list = [os.path.join(opts.content_img_folder, i) for i in self.content_list]

    def __len__(self):
        return len(self.content_list)

    def __getitem__(self, item):
        content_path = self.content_list[item%len(self)]
        style_path = self.style_list[item%len(self)]

        content_img = cv2.imread(content_path, 1)
        style_img = cv2.imread(content_path, 1)

        content_img = torch.from_numpy(content_img.transpose(2,0,1)).float()/self.value_scale
        style_img = torch.from_numpy(style_img.transpose(2,0,1)).float()/self.value_scale

        return {
            "content_img": content_img,
            "style_img": [style_img]
        }