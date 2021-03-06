import copy
import datetime
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from deep_patch_match import VGG19
from tqdm import tqdm

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

class PatchMatch: 
    def __init__(self, a, b, patch_size=3):
        self.a = a
        self.b = b
        self.ah = a.shape[0]
        self.aw = a.shape[1]
        self.bh = b.shape[0]
        self.bw = b.shape[1]
        self.patch_size = patch_size

        self.nnf = np.zeros((self.ah, self.aw, 2)).astype(np.int)  # The NNF
        self.nnd = np.zeros((self.ah, self.aw))  # The NNF distance map
        self.random_global = True
        self.init_nnf()

    def re_init(self, last_level_nnf):
        self.random_global = False
        last_level_nnf *= 2
        self.nnf = cv2.resize(last_level_nnf, dsize=(self.aw, self.ah), interpolation=cv2.INTER_NEAREST)
        for ay in range(self.ah):
            for ax in range(self.aw):
                self.nnd[ay, ax] = self.calc_dist(ay, ax, self.nnf[ay, ax][0], self.nnf[ay, ax][1])

    def init_nnf(self):
        for ay in range(self.ah):
            for ax in range(self.aw):
                by = np.random.randint(self.bh)
                bx = np.random.randint(self.bw)
                self.nnf[ay, ax] = [by, bx]
                self.nnd[ay, ax] = self.calc_dist(ay, ax, by, bx)

    def calc_dist(self, ay, ax, by, bx):
        """
            Measure distance between 2 patches across all channels
            ay : y coordinate of a patch in a
            ax : x coordinate of a patch in a
            by : y coordinate of a patch in b
            bx : x coordinate of a patch in b
        """
        dy0 = dx0 = self.patch_size // 2
        dy1 = dx1 = self.patch_size // 2 + 1
        dy0 = min(ay, by, dy0)
        dy1 = min(self.ah - ay, self.bh - by, dy1)
        dx0 = min(ax, bx, dx0)
        dx1 = min(self.aw - ax, self.bw - bx, dx1)

        dist = np.sum(np.square(self.a[ay - dy0:ay + dy1, ax - dx0:ax + dx1] - self.b[by - dy0:by + dy1, bx - dx0:bx + dx1]))
        dist /= ((dy0 + dy1) * (dx0 + dx1))
        return dist

    def improve_guess(self, ay, ax, by, bx, ybest, xbest, dbest):
        d = self.calc_dist(ay, ax, by, bx)
        if d < dbest:
            ybest, xbest, dbest = by, bx, d
        return ybest, xbest, dbest

    def improve_nnf(self, total_iter=5):
        for iter in tqdm(range(total_iter)):
            if iter % 2:
                ystart, yend, ychange = self.ah - 1, -1, -1
                xstart, xend, xchange = self.aw - 1, -1, -1
            else:
                ystart, yend, ychange = 0, self.ah, 1
                xstart, xend, xchange = 0, self.aw, 1

            for ay in range(ystart, yend, ychange):
                for ax in range(xstart, xend, xchange):
                    ybest, xbest = self.nnf[ay, ax]
                    dbest = self.nnd[ay, ax]

                    # Propagation
                    if 0 <= (ay - ychange) < self.ah:
                        yp, xp = self.nnf[ay - ychange, ax]
                        yp += ychange
                        if 0 <= yp < self.bh:
                            # if ax == 0 and ay==0:
                            ybest, xbest, dbest = self.improve_guess(ay, ax, yp, xp, ybest, xbest, dbest)
                    if 0 <= (ax - xchange) < self.aw:
                        yp, xp = self.nnf[ay, ax - xchange]
                        xp += xchange
                        if 0 <= xp < self.bw:
                            ybest, xbest, dbest = self.improve_guess(ay, ax, yp, xp, ybest, xbest, dbest)

                    # Random search
                    if self.random_global:
                        rand_d = max(self.bh, self.bw)
                        while rand_d >= 1:
                            ymin, ymax = max(ybest - rand_d, 0), min(ybest + rand_d, self.bh)
                            xmin, xmax = max(xbest - rand_d, 0), min(xbest + rand_d, self.bw)
                            yp = np.random.randint(ymin, ymax)
                            xp = np.random.randint(xmin, xmax)
                            ybest, xbest, dbest = self.improve_guess(ay, ax, yp, xp, ybest, xbest, dbest)
                            rand_d = rand_d // 2
                    else:
                        rand_d = 1
                        for rand_x in range(-rand_d, rand_d+1):
                            for rand_y in range(-rand_d, rand_d+1):
                                if rand_x == 0 and rand_y == 0:
                                    continue
                                if ybest + rand_y < 0 or ybest + rand_y > self.bh or xbest + rand_x < 0 or xbest + rand_x > self.bw:
                                    continue
                                # ymin, ymax = max(ybest - rand_d, 0), min(ybest + rand_d, self.bh)
                                # xmin, xmax = max(xbest - rand_d, 0), min(xbest + rand_d, self.bw)
                                yp = ybest + rand_y
                                xp = xbest + rand_x
                                ybest, xbest, dbest = self.improve_guess(ay, ax, yp, xp, ybest, xbest, dbest)
                                rand_d = rand_d // 2

                    self.nnf[ay, ax] = [ybest, xbest]
                    self.nnd[ay, ax] = dbest

    def solve(self, total_iter=8):
        self.improve_nnf(total_iter)

class BidirectNNF(nn.Module):
    def __init__(self, opts, cfg):
        super(BidirectNNF, self).__init__()
        self.opts = opts
        self.cfg = cfg
        self.layers = opts.layers
        self.sizes = opts.patch_sizes
        self.iters = opts.iters
        self.completeness = opts.completeness
        # default: device 0.
        self.VGG19 = VGG19(0)
        self.nnf_AB = None
        self.nnf_BA = None


    def feature_extraction(self, img):
        return self.VGG19.get_features(img.clone(), self.layers)
    
    def compute_guidance_map(self,
                            img_A,
                            img_BP,
                            curr_layer,
                            data_A, 
                            data_A_size, 
                            data_BP, 
                            data_B_size,index=0):
        # print(img_BP)
        print('ITERATION {}'.format(curr_layer))
        b,c,h_A,w_A = data_A.shape
        b,c,h_B,w_B = data_BP.shape
        if curr_layer < 1: 
        # neural-based method.
            time_start = time.time()
            temp_A = normalize(data_A)[0].clone().cuda().view(c,-1).permute(1,0)
            temp_BP = normalize(data_BP)[0].clone().cuda().view(c,-1).permute(1,0)
            norms_A = torch.sum(temp_A**2, dim=1, keepdim=True)
            norms_BP = torch.sum(temp_BP**2, dim=1, keepdim=True)
            norms = (norms_A.expand(h_A*w_A, h_B*w_B) +
                    norms_BP.transpose(0, 1).expand(h_A*w_A, h_B*w_B))
            distances_squared = norms - 2 * temp_A.mm(temp_BP.t())
            distances_squared = torch.abs(distances_squared)
            neural_nnf_forward = torch.argmin(distances_squared, dim=1).view(h_A,w_A, 1)
            neural_nnf_backward = torch.argmin(distances_squared, dim=0).view(h_B,w_B, 1)
            neural_nnf_forward_x = neural_nnf_forward//w_B
            neural_nnf_forward_y = neural_nnf_forward%w_B
            neural_nnf_forward = torch.cat((neural_nnf_forward_x, neural_nnf_forward_y), dim=-1).cpu().numpy()
            neural_nnf_backward_x = neural_nnf_backward//w_A
            neural_nnf_backward_y = neural_nnf_backward%w_A
            neural_nnf_backward = torch.cat((neural_nnf_backward_x, neural_nnf_backward_y), dim=-1).cpu().numpy()
            print("matching_time:", time.time() - time_start)
            data_A = ts2np(normalize(data_A)[0])
            data_BP = ts2np(normalize(data_BP)[0])
        
        else:
            data_A = ts2np(normalize(data_A)[0])
            data_BP = ts2np(normalize(data_BP)[0])
            print("forward nnf matching...")
            PM_forward = PatchMatch(data_A, data_BP)
            if self.nnf_AB is not None:
                PM_forward.re_init(self.nnf_AB)
            PM_forward.solve(self.iters)
            print("backward nnf matching...")
            PM_backward = PatchMatch(data_BP, data_A)
            if self.nnf_BA is not None:
                PM_backward.re_init(self.nnf_BA)
            PM_backward.solve(self.iters)
            neural_nnf_forward = PM_forward.nnf
            neural_nnf_backward = PM_backward.nnf

        img_BP = img_BP[0].numpy().transpose(1,2,0).astype(np.uint8)
        img_BP = cv2.resize(img_BP, (data_B_size[3], data_B_size[2]), cv2.INTER_CUBIC).astype(np.float32)

        print("bidirectional voting...")
        img_AP = bds_vote(img_BP.transpose(2,0,1), neural_nnf_forward, neural_nnf_backward, self.sizes[curr_layer], self.completeness).transpose(1,2,0)
        data_AP_feat = bds_vote(data_BP.transpose(2,0,1), neural_nnf_forward, neural_nnf_backward, self.sizes[curr_layer], self.completeness).transpose(1,2,0)
        data_AP = np2ts(data_AP_feat, 0)
        cv2.imwrite(os.path.join(self.cfg.FOLDER, "{}_guidance_{}.png".format(index, 5-curr_layer)), cv2.cvtColor(img_AP.astype(np.uint8), cv2.COLOR_BGR2RGB))
        # save for the next level.
        # print(self.nnf_AB, PM_forward.nnf)
        self.nnf_AB = neural_nnf_forward
        self.nnf_BA = neural_nnf_backward
        return img_AP, data_AP


def bds_vote(ref, nnf_sr, nnf_rs, patch_size=3, completeness=2):
    """
    Reconstructs an image or feature map by bidirectionaly
    similarity voting
    """

    src_height = nnf_sr.shape[0]
    src_width = nnf_sr.shape[1]
    ref_height = nnf_rs.shape[0]
    ref_width = nnf_rs.shape[1]
    channel = ref.shape[0]

    guide = np.zeros((channel, src_height, src_width))
    weight = np.zeros((src_height, src_width))
    ws = 1 / (src_height * src_width)
    wr = completeness / (ref_height * ref_width)

    # coherence
    # The S->R forward NNF enforces coherence
    for sy in range(src_height):
        for sx in range(src_width):
            ry, rx = nnf_sr[sy, sx]

            dy0 = dx0 = patch_size // 2
            dy1 = dx1 = patch_size // 2 + 1
            dy0 = min(sy, ry, dy0)
            dy1 = min(src_height - sy, ref_height - ry, dy1)
            dx0 = min(sx, rx, dx0)
            dx1 = min(src_width - sx, ref_width - rx, dx1)

            guide[:, sy - dy0:sy + dy1, sx - dx0:sx + dx1] += ws * ref[:, ry - dy0:ry + dy1, rx - dx0:rx + dx1]
            weight[sy - dy0:sy + dy1, sx - dx0:sx + dx1] += ws

    # completeness
    # The R->S backward NNF enforces completeness
    for ry in range(ref_height):
        for rx in range(ref_width):
            sy, sx = nnf_rs[ry, rx]

            dy0 = dx0 = patch_size // 2
            dy1 = dx1 = patch_size // 2 + 1
            dy0 = min(sy, ry, dy0)
            dy1 = min(src_height - sy, ref_height - ry, dy1)
            dx0 = min(sx, rx, dx0)
            dx1 = min(src_width - sx, ref_width - rx, dx1)

            guide[:, sy - dy0:sy + dy1, sx - dx0:sx + dx1] += wr * ref[:, ry - dy0:ry + dy1, rx - dx0:rx + dx1]
            weight[sy - dy0:sy + dy1, sx - dx0:sx + dx1] += wr

    weight[weight == 0] = 1
    guide /= weight
    return guide
