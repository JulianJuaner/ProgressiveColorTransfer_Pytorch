import copy
import datetime
import os
import time
from code.models.nnf import BidirectNNF, blend, normalize, np2ts, ts2np

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_patch_match import (VGG19, avg_vote, init_nnf, propagate,
                              reconstruct_avg, upSample_nnf)
from guided_filter_pytorch.guided_filter import FastGuidedFilter
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


class ProgressiveTransfer(nn.Module):
    def __init__(self, cfg):
        super(ProgressiveTransfer, self).__init__()
        self.opt = cfg.COLORMODEL
        self.cfg = cfg
        self.levels = len(cfg.DEEPMATCH.layers)
        self.nnf_match = BidirectNNF(cfg.DEEPMATCH, cfg)
        self.mse_loss = nn.MSELoss(reduction='none')

        self.smooth_weight = self.opt.smooth_weight
        self.non_local_weight = self.opt.non_local_weight
        self.kmeans_cluster = self.opt.kmeans_cluster
        self.k_nearest_num = self.opt.k_nearest_num

        self.alpha_param = None#nn.Parameter(torch.ones((3, feat_AP.shape[-2], feat_AP.shape[-1])).cuda(), requires_grad=True)
        self.beta_param = None#nn.Parameter(torch.zeros((3, feat_AP.shape[-2], feat_AP.shape[-1])).cuda(), requires_grad=True)
        self.origin_size = None

    def init_input(self, guidance, source, patch_size=7):
        eps = 0.002
        height_s = source.shape[1]
        width_s = source.shape[2]
        for y in range(height_s):
            for x in range(width_s):
                dy0 = dx0 = patch_size // 2
                dy1 = dx1 = patch_size // 2 + 1
                dy0 = min(y, dy0)
                dy1 = min(height_s - y, dy1)
                dx0 = min(x, dx0)
                dx1 = min(width_s - x, dx1)

                patchS = source[:, y - dy0:y + dy1, x - dx0:x + dx1].reshape(3, -1)
                patchG = guidance[:, y - dy0:y + dy1, x - dx0:x + dx1].reshape(3, -1)
                self.alpha_param[:, y, x] = patchG.std(1) / (patchS.std(1) + eps)
                self.beta_param[:, y, x] = patchG.mean(1) - self.alpha_param[:, y, x] * patchS.mean(1)

        self.alpha_param.requires_grad_()
        self.beta_param.requires_grad_()

    def visualize(self, cie_intermidiate):
        alpha_param = F.interpolate(self.alpha_param.detach().unsqueeze(0), size=self.origin_size, mode='bilinear')[0]
        beta_param = F.interpolate(self.beta_param.detach().unsqueeze(0), size=self.origin_size, mode='bilinear')[0]
        intermidiate_img_np = ((cie_intermidiate*alpha_param + beta_param)).cpu().numpy().transpose(1,2,0)
        intermidiate_img_np = np.clip(intermidiate_img_np, 0, 255)
        intermidiate_img_np = cv2.cvtColor(intermidiate_img_np.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return intermidiate_img_np

    def forward(self, data):
        # Save VGG net for style images
        style_img = data["style_img"]

        origin_style = style_img[0].cpu().numpy().transpose(1,2,0).astype(np.uint8)  
        data_BP, data_B_size = self.nnf_match.feature_extraction(style_img.cuda())
        data_B = copy.deepcopy(data_BP)

        # initialize input as the source image.
        intermidiate_img = data["content_img"][0].clone().cuda()
        self.origin_size = intermidiate_img.shape[1:]
        origin_content = intermidiate_img.cpu().numpy().transpose(1,2,0).astype(np.uint8)
        cie_origin_content = torch.FloatTensor(cv2.cvtColor(origin_content.astype(np.uint8), cv2.COLOR_RGB2Lab).transpose(2,0,1)).cuda()

        cv2.imwrite(os.path.join(self.cfg.FOLDER, "content.png"), cv2.cvtColor(origin_content, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(self.cfg.FOLDER, "style.png"), cv2.cvtColor(origin_style, cv2.COLOR_RGB2BGR))
        print(intermidiate_img.shape)
        
        for curr_layer in range(self.levels):
            data_A, data_A_size = self.nnf_match.feature_extraction(intermidiate_img.unsqueeze(0))
            img_AP = intermidiate_img.cpu().numpy().transpose(1,2,0).astype(np.uint8)
            cie_intermidiate = torch.FloatTensor(cv2.cvtColor(img_AP.astype(np.uint8), cv2.COLOR_RGB2Lab).transpose(2,0,1)).cuda()
            img_AP = cv2.resize(img_AP,
                                (data_A_size[curr_layer][3], data_A_size[curr_layer][2]),
                                cv2.INTER_CUBIC).astype(np.float32)
            
            temp_guidance_map, feat_AP = self.nnf_match.compute_guidance_map(intermidiate_img, style_img,
                                        curr_layer, data_A[curr_layer].clone(), data_A_size[curr_layer],
                                        data_BP[curr_layer].clone(), data_B_size[curr_layer])
            
            # color estimation.
            self.alpha_param = nn.Parameter(torch.ones((3, feat_AP.shape[-2], feat_AP.shape[-1])).cuda(), requires_grad=False)
            self.beta_param = nn.Parameter(torch.zeros((3, feat_AP.shape[-2], feat_AP.shape[-1])).cuda(), requires_grad=False)
            cie_dataA = torch.FloatTensor(cv2.cvtColor(img_AP.astype(np.uint8), cv2.COLOR_RGB2Lab).transpose(2,0,1)).cuda()

            # k means local constrain loss term.

            # coarse_feat = data_A[0][0].cpu().numpy().transpose(1,2,0)
            # dataA_kmeans = KMeans(n_clusters=self.kmeans_cluster, n_jobs=1).fit(coarse_feat.reshape(-1, coarse_feat.shape[2]))
            # dataA_kmeans_labels = dataA_kmeans.labels_.reshape(coarse_feat.shape[:2])
            # dataA_kmeans_labels = cv2.resize(dataA_kmeans_labels,
            #                      data_A_size[curr_layer][3], data_A_size[curr_layer][2],
            #                      cv2.INTER_NEAREST)
            # nearest_neighbor = NearestNeighbors(n_neighbors=self.k_nearest_num).fit()

            # print(dataA_kmeans_labels.shape)
            # print(img_AP, cie_dataA)
            cie_guidance = torch.FloatTensor(cv2.cvtColor(temp_guidance_map.astype(np.uint8), cv2.COLOR_RGB2Lab).transpose(2,0,1)).cuda()
            self.init_input(cie_guidance, cie_dataA)

            vis_init = self.visualize(cie_intermidiate)
            cv2.imwrite(os.path.join(self.cfg.FOLDER, "init_inter_{}.png".format(5-curr_layer)), vis_init)
            # print(cie_guidance, cie_dataA, self.mse_loss(cie_dataA, cie_guidance))
            optimizer = torch.optim.Adam([self.alpha_param, self.beta_param], lr=0.000005)
            optimizer.zero_grad()
            feature_error = torch.mean(normalize(self.mse_loss(normalize(feat_AP)[0], normalize(data_A[curr_layer])[0]))[0], dim=1)
            
            print("mean of the feature error:", torch.mean(feature_error))

            for iters in tqdm(range(self.opt.iter*10)):
                def closure():
                    optimizer.zero_grad()
                    intermidiate_result = cie_dataA*self.alpha_param + self.beta_param
                    # intermidiate_result = F.sigmoid(intermidiate_result/255)*255
                    # error_one: guidance limitation.
                    e_d = 4**(4-curr_layer)*torch.mean((1-feature_error)*(self.mse_loss(intermidiate_result, cie_guidance)))
                    
                    # error_two: smoothness limitation: like total variation.
                    e_up = self.mse_loss(self.alpha_param, torch.roll(self.alpha_param, shifts=-1, dims=1)) + \
                        self.mse_loss(self.beta_param, torch.roll(self.beta_param, shifts=-1, dims=1))
                    e_up *= 1/(torch.pow(torch.abs(cie_dataA[0] - torch.roll(cie_dataA[0], shifts=-1, dims=0)), 1.2) + 1e-4)
                    e_down = self.mse_loss(self.alpha_param, torch.roll(self.alpha_param, shifts=1, dims=1)) + \
                        self.mse_loss(self.beta_param, torch.roll(self.beta_param, shifts=1, dims=1))
                    e_down *= 1/(torch.pow(torch.abs(cie_dataA[0] - torch.roll(cie_dataA[0], shifts=1, dims=0)), 1.2) + 1e-4)
                    e_left = self.mse_loss(self.alpha_param, torch.roll(self.alpha_param, shifts=-1, dims=2)) + \
                        self.mse_loss(self.beta_param, torch.roll(self.beta_param, shifts=-1, dims=2))
                    e_left *= 1/(torch.pow(torch.abs(cie_dataA[0] - torch.roll(cie_dataA[0], shifts=-1, dims=1)), 1.2) + 1e-4)
                    e_right = self.mse_loss(self.alpha_param, torch.roll(self.alpha_param, shifts=1, dims=2)) + \
                        self.mse_loss(self.beta_param, torch.roll(self.beta_param, shifts=1, dims=2))
                    e_right *= 1/(torch.pow(torch.abs(cie_dataA[0] - torch.roll(cie_dataA[0], shifts=1, dims=1)), 1.2) + 1e-4)

                    e_l = torch.mean(self.smooth_weight*(e_up + e_down + e_left + e_right))
                    # print(e_l.item(), e_d.item())
                    Loss = 1*e_d + e_l*0.1
                    Loss.backward()
                    return Loss

                # Loss = closure()
                
                # Loss.backward()
                optimizer.step(closure)

            # print(torch.from_numpy(img_AP.transpose(2,0,1)).unsqueeze(0).shape, self.alpha_param.detach().unsqueeze(0).cpu().shape, cie_intermidiate.unsqueeze(0).shape)
            # alpha_param = FastGuidedFilter(1, eps=1e-08)(torch.from_numpy(img_AP.transpose(2,0,1)).unsqueeze(0).cuda(),
            #                                  self.alpha_param.detach().unsqueeze(0).cuda(),
            #                                  cie_intermidiate.unsqueeze(0).cuda()).squeeze()
            # beta_param = FastGuidedFilter(1, eps=1e-08)(torch.from_numpy(img_AP.transpose(2,0,1)).unsqueeze(0).cuda(),
            #                                  self.beta_param.detach().unsqueeze(0).cuda(),
            #                                  cie_intermidiate.unsqueeze(0).cuda()).squeeze()

            alpha_param = F.interpolate(self.alpha_param.detach().unsqueeze(0), size=self.origin_size, mode='bilinear')[0]
            beta_param = F.interpolate(self.beta_param.detach().unsqueeze(0), size=self.origin_size, mode='bilinear')[0]

            # intermidiate_img_np = ((F.sigmoid(cie_intermidiate*alpha_param + beta_param))).cpu().numpy().transpose(1,2,0)*255
            intermidiate_img_np = (cie_intermidiate*alpha_param + beta_param).cpu().numpy().transpose(1,2,0)
            intermidiate_img_np = np.clip(intermidiate_img_np, 0, 255)
            intermidiate_img_np = cv2.cvtColor(intermidiate_img_np.astype(np.uint8), cv2.COLOR_LAB2BGR)
            cv2.imwrite(os.path.join(self.cfg.FOLDER, "inter_res_{}.png".format(5-curr_layer)), intermidiate_img_np)
            intermidiate_img_ = torch.FloatTensor(intermidiate_img_np.transpose(2,0,1)).cuda()
            intermidiate_img = (intermidiate_img + intermidiate_img_) /2

        res_dict = dict()
        return res_dict

