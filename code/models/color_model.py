from code.models.nnf import BidirectNNF, normalize, ts2np, np2ts, blend
from deep_patch_match import VGG19, init_nnf, upSample_nnf, avg_vote, propagate, reconstruct_avg

import copy
import time
import datetime
import torch
import torch.nn as nn


class ProgressiveTransfer(nn.Module):
    def __init__(self, cfg):
        super(ProgressiveTransfer, self).__init__()
        self.cfg = cfg
        self.levels = len(cfg.DEEPMATCH.layers)
        self.nnf_match = BidirectNNF(cfg.DEEPMATCH, cfg)
        self.color_estimate = ColorModel(cfg.COLORMODEL, cfg)

    def forward(self, data):
        # Save VGG net for style images
        style_img = data["style_img"]        
        data_BP, data_B_size = self.nnf_match.feature_extraction(style_img.cuda())
        data_B = copy.deepcopy(data_BP)

        # initialize input as the source image.
        intermidiate_img = data["content_img"].clone().cuda()
        print(intermidiate_img.shape)
        for curr_layer in range(self.levels):
            data_A, data_A_size = self.nnf_match.feature_extraction(intermidiate_img)
            temp_guidance_map = self.nnf_match.compute_guidance_map(intermidiate_img, style_img,
                                        curr_layer, data_A[curr_layer], data_A_size[curr_layer],
                                        data_BP[curr_layer], data_B_size[curr_layer])
            

        res_dict = dict()
        return res_dict

class ColorModel(nn.Module):
    def __init__(self, opts, cfg):
        super(ColorModel, self).__init__()
        self.opts = opts
        self.cfg = cfg
