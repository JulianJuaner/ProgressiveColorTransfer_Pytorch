from code.models.nnf import BidirectNNF

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
        style_feat_list = []
        style_size_list = []
        style_list = data["style_img"]
        for style_img in style_list:
            feat, size = self.nnf_match.feature_extraction(style_img.cuda())
            style_feat_list.append(feat)
            style_size_list.append(size)

        print(style_feat_list[0][0].shape)

        for ref in range(len(style_list)):
            for i in range(len(self.levels)):
                temp_guidance_map = self.nnf_match.compute_guidance_map(data["content_img"].cuda(),
                                        style_feat_list[ref][i])

        res_dict = dict()
        return res_dict

class ColorModel(nn.Module):
    def __init__(self, opts, cfg):
        super(ColorModel, self).__init__()
        self.opts = opts
        self.cfg = cfg
