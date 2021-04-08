import copy
import datetime
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from deep_patch_match import (VGG19, avg_vote, init_nnf, propagate,
                              reconstruct_avg, upSample_nnf)


def ts2np(x):
    x = x.squeeze(0)
    x = x.cpu().numpy()
    x = x.transpose(1,2,0)
    return x

def np2ts(x, device):
    x = x.transpose(2,0,1)
    x = torch.from_numpy(x)
    x = x.unsqueeze(0)
    x = x.to(device)
    return x



def blend(response, f_a, r_bp, alpha=0.8, tau=0.05):
    """
    :param response:
    :param f_a: feature map (either F_A or F_BP)
    :param r_bp: reconstructed feature (R_BP or R_A)
    :param alpha: scalar balance the ratio of content and style in new feature map
    :param tau: threshold, default: 0.05 (suggested in paper)
    :return: (f_a*W + r_bp*(1-W)) where W=alpha*(response>tau)

    Following the official implementation, I replace the sigmoid function (stated in paper) with indicator function
    """
    weight = (response > tau).type(f_a.type()) * alpha
    weight = weight.expand(1, f_a.size(1), weight.size(2), weight.size(3))

    f_ap = f_a*weight + r_bp*(1. - weight)
    return f_ap


def normalize(feature_map):
    """

    :param feature_map: either F_a or F_bp
    :return:
    normalized feature map
    response
    """
    # SUMMATION: 1, C, H, W: ALONG DIM 1: 1, H, W
    response = torch.sum(feature_map*feature_map, dim=1, keepdim=True)
    normed_feature_map = feature_map/torch.sqrt(response)

    # response should be scaled to (0, 1)
    response = (response-torch.min(response))/(torch.max(response)-torch.min(response))
    return  normed_feature_map, response


class BidirectNNF(nn.Module):
    def __init__(self, opts, cfg):
        super(BidirectNNF, self).__init__()
        self.opts = opts
        self.cfg = cfg
        self.layers = opts.layers
        if opts.weight == 2:
            self.weights = [1.0, 0.8, 0.7, 0.6, 0.1, 0.0]
        else:
            self.weights = [1.0, 0.9, 0.8, 0.7, 0.2, 0.0]
        self.sizes = opts.patch_sizes
        self.iters = opts.iters
        self.rangee = opts.rangee
        # default: device 0.
        self.VGG19 = VGG19(0)


    def feature_extraction(self, img):
        return self.VGG19.get_features(img.clone(), self.layers)
    
    def compute_guidance_map(self,
                            img_A,
                            img_BP,
                            curr_layer,
                            data_A, 
                            data_A_size, 
                            data_BP, 
                            data_B_size):
        # print(img_BP)
        img_BP = img_BP[0].numpy().transpose(1,2,0).astype(np.uint8)
        img_BP = cv2.resize(img_BP, (data_B_size[3], data_B_size[2]), cv2.INTER_CUBIC).astype(np.float32)
        ann_AB = init_nnf(data_A_size[2:], data_B_size[2:])
        ann_BA = init_nnf(data_B_size[2:], data_A_size[2:])
        data_AP = copy.deepcopy(data_A)
        data_B = copy.deepcopy(data_BP)

        Ndata_A, response_A = normalize(data_A)
        Ndata_BP, response_BP = normalize(data_BP)

        data_AP = blend(response_A, data_A, data_AP, self.weights[curr_layer])
        data_B = blend(response_BP, data_BP, data_B, self.weights[curr_layer])

        Ndata_AP, _ = normalize(data_AP)
        Ndata_B, _ = normalize(data_B)

        # NNF search
        print("- NNF search for self.ann_AB")
        start_time_2 = time.time()
        ann_AB, _ = propagate(ann_AB, ts2np(Ndata_A), ts2np(Ndata_AP), ts2np(Ndata_B), ts2np(Ndata_BP), self.sizes[curr_layer],
                              self.iters, self.rangee[curr_layer])
        print("\tElapse: "+str(datetime.timedelta(seconds=time.time()- start_time_2))[:-7])

        # print("- NNF search for self.ann_BA")
        # start_time_2 = time.time()
        # ann_BA, _ = propagate(ann_BA, ts2np(Ndata_BP), ts2np(Ndata_B), ts2np(Ndata_AP), ts2np(Ndata_A), self.sizes[curr_layer],
        #                       self.iters, self.rangee[curr_layer])
        # # print(ann_BA)
        # print("\tElapse: "+str(datetime.timedelta(seconds=time.time()- start_time_2))[:-7])

       
        # ann_AB_backward = np.ones((data_A_size[2], data_A_size[3], 2))*-1

        # for i in range(ann_BA.shape[0]):
        #     for j in range(ann_BA.shape[1]):
        #         ann_AB_backward[ann_BA[i]][ann_BA[j]] = np.array([i, j])

        # img_AP_b = reconstruct_avg(ann_AB_backward, img_BP, self.sizes[curr_layer], data_A_size[2:], data_B_size[2:])
        # img_AP = cv2.cvtColor((img_AP/3 + img_AP_b*2/3).astype(np.uint8), cv2.COLOR_BGR2RGB)
        
        data_AP_np = avg_vote(ann_AB, ts2np(data_BP), self.sizes[curr_layer], data_A_size[2:],
                              data_B_size[2:])
        data_AP = np2ts(data_AP_np, 0)

        img_AP = reconstruct_avg(ann_AB, img_BP, self.sizes[curr_layer], data_A_size[2:], data_B_size[2:])
        # cv2.cvtColor(img_AP.astype(np.uint8), cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(self.cfg.FOLDER, "guidance_{}.png".format(5-curr_layer)), cv2.cvtColor(img_AP.astype(np.uint8), cv2.COLOR_BGR2RGB))

        return img_AP, data_AP


def BDS_vote(nnf_AB, nnf_BA, A_size, B_size, weight):
    assert img.shape[0] == B_size[0] and img.shape[1] == B_size[1], "[{},{}], [{},{}]".format(img.shape[0],
                                                                                              img.shape[1], B_size[0],
                                                                                              B_size[1])
    final = np.zeros(list(A_size) + [img.shape[2], ])

    ah, aw = A_size
    bh, bw = B_size
    for ay in range(A_size[0]):
        for ax in range(A_size[1]):

            count = 0
            for dy in range(-(patch_size // 2), (patch_size // 2 + 1)):
                for dx in range(-(patch_size // 2), (patch_size // 2 + 1)):

                    if ((ax + dx) < aw and (ax + dx) >= 0 and (ay + dy) < ah and (ay + dy) >= 0):
                        by, bx = nnf[ay + dy, ax + dx]

                        if ((bx - dx) < bw and (bx - dx) >= 0 and (by - dy) < bh and (by - dy) >= 0):
                            count += 1
                            final[ay, ax, :] += img[by - dy, bx - dx, :]

            if count > 0:
                final[ay, ax] /= count

    return final
