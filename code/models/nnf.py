from deep_patch_match import VGG19, init_nnf, upSample_nnf, avg_vote, propagate, reconstruct_avg
import torch
import torch.nn as nn


class BidirectNNF(nn.Module):
    def __init__(self, opts, cfg):
        super(BidirectNNF, self).__init__()
        self.opts = opts
        self.cfg = cfg
        self.layers = opts.layers
        # default device 0.
        self.VGG19 = VGG19(0)

    def feature_extraction(self, img):
        return self.VGG19.get_features(img, self.layers)
    
    def compute_guidance_map(self, sytle_feat, content_img):
        pass