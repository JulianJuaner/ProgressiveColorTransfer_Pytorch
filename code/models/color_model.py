from code.models.nnf import BidirectNNF, normalize, ts2np, np2ts, blend
from deep_patch_match import VGG19, init_nnf, upSample_nnf, avg_vote, propagate, reconstruct_avg
import os
import cv2
import numpy as np
import copy
import time
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProgressiveTransfer(nn.Module):
    def __init__(self, cfg):
        super(ProgressiveTransfer, self).__init__()
        self.opt = cfg.COLORMODEL
        self.cfg = cfg
        self.levels = len(cfg.DEEPMATCH.layers)
        self.nnf_match = BidirectNNF(cfg.DEEPMATCH, cfg)
        self.mse_loss = nn.MSELoss(reduction='none')
        self.smooth_weight = self.opt.smooth_weight

    def forward(self, data):
        # Save VGG net for style images
        style_img = data["style_img"]
        origin_style = style_img[0].cpu().numpy().transpose(1,2,0).astype(np.uint8)  
        data_BP, data_B_size = self.nnf_match.feature_extraction(style_img.cuda())
        data_B = copy.deepcopy(data_BP)

        # initialize input as the source image.
        intermidiate_img = data["content_img"][0].clone().cuda()
        origin_size = intermidiate_img.shape[1:]
        origin_content = intermidiate_img.cpu().numpy().transpose(1,2,0).astype(np.uint8)
        cie_origin_content = torch.FloatTensor(cv2.cvtColor(origin_content.astype(np.uint8), cv2.COLOR_BGR2Lab).transpose(2,0,1)).cuda()/100
        cv2.imwrite(os.path.join(self.cfg.FOLDER, "content.png"), cv2.cvtColor(origin_content, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(self.cfg.FOLDER, "style.png"), cv2.cvtColor(origin_style, cv2.COLOR_RGB2BGR))
        print(intermidiate_img.shape)
        
        for curr_layer in range(self.levels):
            data_A, data_A_size = self.nnf_match.feature_extraction(intermidiate_img.unsqueeze(0))
            img_AP = intermidiate_img.cpu().numpy().transpose(1,2,0).astype(np.uint8)
            cie_intermidiate = torch.FloatTensor(cv2.cvtColor(img_AP.astype(np.uint8), cv2.COLOR_BGR2Lab).transpose(2,0,1)).cuda()/100
            img_AP = cv2.resize(img_AP,
                                (data_A_size[curr_layer][3], data_A_size[curr_layer][2]),
                                cv2.INTER_CUBIC).astype(np.float32)
            
            temp_guidance_map, feat_AP = self.nnf_match.compute_guidance_map(intermidiate_img, style_img,
                                        curr_layer, data_A[curr_layer], data_A_size[curr_layer],
                                        data_BP[curr_layer], data_B_size[curr_layer])
            
            # color estimation.
            alpha_param = nn.Parameter(torch.ones((3, feat_AP.shape[-2], feat_AP.shape[-1])).cuda(), requires_grad=True)
            beta_param = nn.Parameter(torch.zeros((3, feat_AP.shape[-2], feat_AP.shape[-1])).cuda(), requires_grad=True)
            cie_dataA = torch.FloatTensor(cv2.cvtColor(img_AP.astype(np.uint8), cv2.COLOR_BGR2Lab).transpose(2,0,1)).cuda()/100
            # print(img_AP, cie_dataA)
            cie_guidance = torch.FloatTensor(cv2.cvtColor(temp_guidance_map.astype(np.uint8), cv2.COLOR_BGR2Lab).transpose(2,0,1)).cuda()/100
            # print(cie_guidance, cie_dataA, self.mse_loss(cie_dataA, cie_guidance))
            optimizer = torch.optim.LBFGS([alpha_param, beta_param], lr=3e-6)
            optimizer.zero_grad()
            feature_error = torch.mean(normalize(self.mse_loss(feat_AP, data_A[curr_layer]))[0], dim=1)
            
            print(feature_error.shape)
            for iters in range(self.opt.iter):
                def closure():
                    intermidiate_result = cie_dataA*alpha_param + beta_param
                    # error_one: guidance limitation.
                    e_d = 4**(4-curr_layer)*torch.mean((1-feature_error)*(self.mse_loss(intermidiate_result, cie_guidance)))
                    
                    # error_two: smoothness limitation: like total variation.
                    e_up = self.mse_loss(alpha_param, torch.roll(alpha_param, shifts=-1, dims=1)) + \
                        self.mse_loss(beta_param, torch.roll(beta_param, shifts=-1, dims=1))
                    e_up *= 1/(torch.pow(torch.abs(cie_dataA[0] - torch.roll(cie_dataA[0], shifts=-1, dims=0)), 1.2) + 1e-4)
                    e_down = self.mse_loss(alpha_param, torch.roll(alpha_param, shifts=1, dims=1)) + \
                        self.mse_loss(beta_param, torch.roll(beta_param, shifts=1, dims=1))
                    e_down *= 1/(torch.pow(torch.abs(cie_dataA[0] - torch.roll(cie_dataA[0], shifts=1, dims=0)), 1.2) + 1e-4)
                    e_left = self.mse_loss(alpha_param, torch.roll(alpha_param, shifts=-1, dims=2)) + \
                        self.mse_loss(beta_param, torch.roll(beta_param, shifts=-1, dims=2))
                    e_left *= 1/(torch.pow(torch.abs(cie_dataA[0] - torch.roll(cie_dataA[0], shifts=-1, dims=1)), 1.2) + 1e-4)
                    e_right = self.mse_loss(alpha_param, torch.roll(alpha_param, shifts=1, dims=2)) + \
                        self.mse_loss(beta_param, torch.roll(beta_param, shifts=1, dims=2))
                    e_right *= 1/(torch.pow(torch.abs(cie_dataA[0] - torch.roll(cie_dataA[0], shifts=1, dims=1)), 1.2) + 1e-4)

                    e_l = torch.mean(self.smooth_weight*(e_up + e_down + e_left + e_right))
                    Loss = e_d + e_l
                    Loss.backward(retain_graph=True)
                    return Loss

                optimizer.step(closure)

            alpha_param = F.interpolate(alpha_param.detach().unsqueeze(0), size=origin_size, mode='bilinear')[0]
            beta_param = F.interpolate(beta_param.detach().unsqueeze(0), size=origin_size, mode='bilinear')[0]

            intermidiate_img_np = (100*(cie_intermidiate*alpha_param + beta_param)).cpu().numpy().transpose(1,2,0).astype(np.uint8)
            intermidiate_img_np = cv2.cvtColor(intermidiate_img_np, cv2.COLOR_LAB2BGR)
            cv2.imwrite(os.path.join(self.cfg.FOLDER, "inter_res_{}.png".format(5-curr_layer)), cv2.cvtColor(intermidiate_img_np, cv2.COLOR_BGR2RGB))
            intermidiate_img = torch.FloatTensor(intermidiate_img_np.transpose(2,0,1)).cuda()

        res_dict = dict()
        return res_dict

