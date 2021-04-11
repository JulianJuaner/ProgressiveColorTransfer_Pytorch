# 读代码开始。
import matplotlib as plt
import numpy as np
import torch
import os
import cv2
from PIL import Image
from skimage import color
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from torch import nn, optim
from torchvision import models, transforms, utils

USE_CUDA = True  # or False if you don't have CUDA
FEATURE_IDS = [1, 6, 11, 20, 29]
LEFT_SHIFT = (1, 2, 0)
RIGHT_SHIFT = (2, 0, 1)
root = "/home/jlzhang/projects/misc/Neural_Color_Transfer/"
imgS_path = root + 'image/source_succ.jpg'
imgR_path = root + 'image/target_succ.jpg'

# Checking
origS = Image.open(imgS_path).convert("RGB")
laborigS = Image.fromarray(color.rgb2lab(np.asarray(origS)).astype(np.uint8))
# imshow(origS)
# Checking
origR = Image.open(imgR_path).convert("RGB")
# imshow(origR)

def image_loader(img_path, flip=False):
    img = Image.open(img_path).convert("RGB")
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_tensor = data_transforms(img)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


def tsshow(img_tensor):
    img_np = img_tensor.squeeze().numpy().transpose(LEFT_SHIFT)
    # imshow(img_np)  # , interpolation = 'nearest')

imgS = image_loader(imgS_path, flip=False)
imgR = image_loader(imgR_path, flip=False)

imgS_np = imgS.squeeze().numpy().transpose(LEFT_SHIFT)
imgR_np = imgR.squeeze().numpy().transpose(LEFT_SHIFT)

# Checking
print(imgS.size())  # (1, 3, SHeight, SWidth)
print(imgR.size())  # (1, 3, RHeight, RWidth)
print(imgS.dtype, imgR.dtype, "\n")  # torch.float32

print(imgS_np.shape)  # (SHeight, SWidth, 3)
print(imgR_np.shape, "\n")  # (RHeight, RWidth, 3)

# Verifying normalization
print("Original S's mean:", np.asarray(origS).mean(axis=(0, 1)))
print("Original S's std:", np.asarray(origS).std(axis=(0, 1)))
print("Normalized S's mean:", imgS_np.mean(axis=(0, 1)))
print("Normalized S's std:", imgS_np.std(axis=(0, 1)), "\n")

print("Original R's mean:", np.asarray(origR).mean(axis=(0, 1)))
print("Original R's std:", np.asarray(origR).std(axis=(0, 1)))
print("Normalized R's mean:", imgR_np.mean(axis=(0, 1)))
print("Normalized R's std:", imgR_np.std(axis=(0, 1)))

# Checking
tsshow(imgS)

# Checking
tsshow(imgR)

class FeatureExtractor(nn.Sequential):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

    def add_layer(self, name, layer):
        self.add_module(name, layer)

    def forward(self, x, feature_id):
        for idx, module in enumerate(self._modules):
            x = self._modules[module](x)
            if idx == feature_id:
                return x

vgg_temp = models.vgg19(pretrained=True).features
model = FeatureExtractor()  # The new Feature Extractor module network

conv_counter = 1
relu_counter = 1
block_counter = 1

for i, layer in enumerate(list(vgg_temp)):
    if isinstance(layer, nn.Conv2d):
        name = "conv_" + str(block_counter) + "_" + str(conv_counter)
        conv_counter += 1
        model.add_layer(name, layer)

    if isinstance(layer, nn.ReLU):
        name = "relu_" + str(block_counter) + "_" + str(relu_counter)
        relu_counter += 1
        model.add_layer(name, layer)

    if isinstance(layer, nn.MaxPool2d):
        name = "pool_" + str(block_counter) 
        relu_counter = conv_counter = 1
        block_counter += 1
        model.add_layer(name, nn.AvgPool2d(2, 2))  # Is nn.AvgPool2d(2, 2) better than nn.MaxPool2d?

if USE_CUDA:
    model.cuda('cuda:0')

# Checking
print(model)
print([list(model._modules)[idx] for idx in FEATURE_IDS])

def get_feature(img_tensor, feature_id):
    if USE_CUDA:
        img_tensor = img_tensor.cuda('cuda:0')

    feature_tensor = model(img_tensor, feature_id)
    feature = feature_tensor.data.squeeze().cpu().numpy().transpose(LEFT_SHIFT)
    return feature


def normalize(feature):
    return feature / np.linalg.norm(feature, ord=2, axis=2, keepdims=True)

feat5S = get_feature(imgS, FEATURE_IDS[4])
feat5R = get_feature(imgR, FEATURE_IDS[4])
feat5S_norm = normalize(feat5S)
feat5R_norm = normalize(feat5R)

# Checking
print(feat5S.shape)
print(feat5R.shape)

# (IN PROGRESS)
# EXPERIMENTAL
class DeepDream:
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter, im_path):
        self.model = model
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        self.image = image_loader(im_path, flip=False)

        if USE_CUDA:
            self.model.cuda('cuda:0')
            self.image = self.image.cuda('cuda:0')

        self.image.requires_grad_()
        # Hook the layers to get result of the convolution
        self.hook_layer()

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Get the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]

        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)

    def dream(self):
        # Define optimizer for the image
        # Earlier layers need higher learning rates to visualize whereas lower layers need less
        optimizer = optim.SGD([self.image], lr=12, weight_decay=1e-4)
        # optimizer = optim.Adam([self.image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 251):
            optimizer.zero_grad()
            # Assign image to a variable to move forward in the model
            x = self.image
            for index, layer in enumerate(self.model):
                # Forward
                x = layer(x)
                # Only need to forward until the selected layer is reached
                if index == self.selected_layer:
                    break
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            if i % 10 == 0:
                print("Iteration:", str(i) + "/250", "Loss: {0:.2f}".format(loss.data))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()

# # (IN PROGRESS)
# # EXPERIMENTAL
# cnn_layer = FEATURE_IDS[4]
# filter_pos = 94
# dd = DeepDream(vgg_temp, cnn_layer, filter_pos, imgS_path)
# # This operation can also be done without Pytorch hooks
# # See layer visualisation for the implementation without hooks
# dd.dream()

# dd_image_np = dd.image.data.squeeze().cpu().numpy().transpose(LEFT_SHIFT)
# # imshow(dd_image_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406])

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
        self.init_nnf()

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
        for iter in range(total_iter):
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
                            ybest, xbest, dbest = self.improve_guess(ay, ax, yp, xp, ybest, xbest, dbest)
                    if 0 <= (ax - xchange) < self.aw:
                        yp, xp = self.nnf[ay, ax - xchange]
                        xp += xchange
                        if 0 <= xp < self.bw:
                            ybest, xbest, dbest = self.improve_guess(ay, ax, yp, xp, ybest, xbest, dbest)

                    # Random search
                    rand_d = max(self.bh, self.bw)
                    while rand_d >= 1:
                        ymin, ymax = max(ybest - rand_d, 0), min(ybest + rand_d, self.bh)
                        xmin, xmax = max(xbest - rand_d, 0), min(xbest + rand_d, self.bw)
                        yp = np.random.randint(ymin, ymax)
                        xp = np.random.randint(xmin, xmax)
                        ybest, xbest, dbest = self.improve_guess(ay, ax, yp, xp, ybest, xbest, dbest)
                        rand_d = rand_d // 2

                    self.nnf[ay, ax] = [ybest, xbest]
                    self.nnd[ay, ax] = dbest
            
            print("iteration:", str(iter + 1) + "/" + str(total_iter))

    def solve(self):
        self.improve_nnf(total_iter=8)

map5SR = PatchMatch(feat5S_norm, feat5R_norm)  # S -> R
map5RS = PatchMatch(feat5R_norm, feat5S_norm)  # R -> S
map5SR.solve()
print()
map5RS.solve()

# Checking
print(map5SR.nnf.shape)
print(map5SR.nnd.shape, "\n")

print(map5RS.nnf.shape)
print(map5RS.nnd.shape)

def image_to_tensor(img, img_transforms=None):
    if img_transforms is None:
        img_transforms = list()
    data_transforms = transforms.Compose(img_transforms + [
        transforms.ToTensor(),
    ])
    img_tensor = data_transforms(img)
    return img_tensor


def resize_img(img, size):
    return image_to_tensor(img, [transforms.Resize(size)])

imgS_resized = resize_img(origS, feat5S.shape[:2])
imgR_resized = resize_img(origR, feat5R.shape[:2])

# Checking
print(imgS_resized.size())
print(imgR_resized.size())

# Checking
tsshow(imgS_resized)

# Checking
tsshow(imgR_resized)

def bds_vote(ref, nnf_sr, nnf_rs, patch_size=3):
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
    wr = 1 / (ref_height * ref_width)

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
print(type(imgR_resized), type(map5SR.nnf),type(map5RS.nnf))
imgG = bds_vote(imgR_resized.numpy(), map5SR.nnf, map5RS.nnf)
feat5G = bds_vote(feat5R.transpose(RIGHT_SHIFT), map5SR.nnf, map5RS.nnf).transpose(LEFT_SHIFT)
feat5G_norm = normalize(feat5G)

# Checking
print(imgG.shape, "==", imgR_resized.size()[0], map5SR.nnf.shape[:2])
print(feat5G.shape, "==", map5SR.nnf.shape[:2], feat5R.shape[2])

# Checking
# imshow(imgG.transpose(LEFT_SHIFT))

kmeans = KMeans(n_clusters=5, n_jobs=1).fit(feat5S.reshape(-1, feat5S.shape[2]))
kmeans_labels = kmeans.labels_.reshape(feat5S.shape[:2])

labS = color.rgb2lab(imgS_resized.numpy().transpose(LEFT_SHIFT))
labG = color.rgb2lab(imgG.transpose(LEFT_SHIFT))

# Checking
print(labS.shape)
print(labG.shape)

class LocalColorTransfer:
    def __init__(self, s, g, featS_norm, featG_norm, L=0, kmeans_ratio=1, patch_size=5):
        self.source = torch.from_numpy(s).float()
        self.guide = torch.from_numpy(g).float()
        self.featS_norm = torch.from_numpy(featS_norm).float()
        self.featG_norm = torch.from_numpy(featG_norm).float()
        self.height = s.shape[0]
        self.width = s.shape[1]
        self.channel = s.shape[2]
        self.patch_size = patch_size
        self.L = 0
        self.paramA = torch.zeros(s.shape)
        self.paramB = torch.zeros(s.shape)
        self.sub = torch.ones(*s.shape[:2], 1)

        self.kmeans_labels = np.zeros(s.shape[:2]).astype(np.int32)
        self.kmeans_ratio = kmeans_ratio

        if USE_CUDA:
            self.source = self.source.cuda('cuda:0')
            self.guide = self.guide.cuda('cuda:0')
            self.featS_norm = self.featS_norm.cuda('cuda:0')
            self.featG_norm = self.featG_norm.cuda('cuda:0')
            self.paramA = self.paramA.cuda('cuda:0')
            self.paramB = self.paramB.cuda('cuda:0')
            self.sub = self.sub.cuda('cuda:0')
        self.init_params()

    def init_params(self):
        """
            Initialize a and b from source and guidance using mean and std
        """
        eps = 0.002
        for y in range(self.height):
            for x in range(self.width):
                dy0 = dx0 = self.patch_size // 2
                dy1 = dx1 = self.patch_size // 2 + 1
                dy0 = min(y, dy0)
                dy1 = min(self.height - y, dy1)
                dx0 = min(x, dx0)
                dx1 = min(self.width - x, dx1)

                patchS = self.source[y - dy0:y + dy1, x - dx0:x + dx1].reshape(-1, self.channel)
                patchG = self.guide[y - dy0:y + dy1, x - dx0:x + dx1].reshape(-1, self.channel)
                self.paramA[y, x] = patchG.std(0) / (patchS.std(0) + eps)
                self.paramB[y, x] = patchG.mean(0) - self.paramA[y, x] * patchS.mean(0)
                self.sub[y, x, 0] += self.patch_size ** 2 - (dy0 + dy1) * (dx0 + dx1)

                y_adj = min(y // self.kmeans_ratio, kmeans_labels.shape[0] - 1)
                x_adj = min(x // self.kmeans_ratio, kmeans_labels.shape[1] - 1)
                self.kmeans_labels[y, x] = kmeans_labels[y_adj, x_adj]
        self.paramA.requires_grad_()
        self.paramB.requires_grad_()

    def visualize(self):
        transfered = self.paramA * self.source + self.paramB
        # imshow(transfered.data.cpu().numpy().astype(np.float64))
        # # imshow(color.lab2rgb(transfered.data.cpu().numpy().astype(np.float64)))

    def loss_d(self):
        error = torch.pow(self.featS_norm - self.featG_norm, 2).sum(2)
        transfered = self.paramA * self.source + self.paramB
        term1 = (1 - error/4)*(4 ** (4 - self.L))
        term2 = torch.pow(transfered - self.guide, 2).sum(2)
        loss_d = torch.mean(term1 * term2)

        return loss_d

    def loss_l(self):
        patchS_stack = self.source.unsqueeze(2).repeat(1, 1, self.patch_size ** 2, 1)  # (self.height, self.width, 9, self.channel)
        patchA_stack = self.paramA.unsqueeze(2).repeat(1, 1, self.patch_size ** 2, 1)
        patchB_stack = self.paramB.unsqueeze(2).repeat(1, 1, self.patch_size ** 2, 1)
        for y in range(self.height):
            for x in range(self.width):
                dy0 = dx0 = self.patch_size // 2
                dy1 = dx1 = self.patch_size // 2 + 1
                dy0 = min(y, dy0)
                dy1 = min(self.height - y, dy1)
                dx0 = min(x, dx0)
                dx1 = min(self.width - x, dx1)

                patchS_stack[y, x, :((dy0 + dy1) * (dx0 + dx1))] = self.source[y - dy0:y + dy1, x - dx0:x + dx1].reshape(-1, self.channel)
                patchA_stack[y, x, :((dy0 + dy1) * (dx0 + dx1))] = self.paramA[y - dy0:y + dy1, x - dx0:x + dx1].reshape(-1, self.channel)
                patchB_stack[y, x, :((dy0 + dy1) * (dx0 + dx1))] = self.paramB[y - dy0:y + dy1, x - dx0:x + dx1].reshape(-1, self.channel)

        patchSD = torch.norm(self.source.unsqueeze(2) - patchS_stack, 2, 3).exp()
        wgt = patchSD / (patchSD.sum(2, keepdim=True) - self.sub)
        # Getting norm term
        term1 = torch.pow(self.paramA.unsqueeze(2) - patchA_stack, 2).sum(3)
        term2 = torch.pow(self.paramB.unsqueeze(2) - patchB_stack, 2).sum(3)
        term3 = term1 + term2
        loss_l = torch.sum(wgt * term3, 2).mean()

        return loss_l

        """
            if y == 0:
                if x == 0:
                    allA = patchA[0 ,0].view(3, 1, 1)  # left up corner
                elif x == self.width - 1:
                    allA = patchA[0, 1].view(3, 1, 1)  # right up corner
                else:
                    allA = patchA[0, 1].view(3, 1, 1)
            elif y == self.height - 1:
                if x == 0:
                    allA = patchA[1, 0].view(3, 1, 1)  # left down corner
                elif x == self.width - 1:
                    allA = patchA[1, 1].view(3, 1, 1)  # right down corner
                else:
                    allA = patchA[1, 1].view(3, 1, 1)
            else:
                if x == 0:
                    allA = patchA[1, 0].view(3, 1, 1)  # left middle
                elif x == self.width - 1:
                    allA = patchA[1, 1].view(3, 1, 1)  # right middle
                else:
                    allA = patchA[1, 1].view(3, 1, 1)  # middle

            멍청한 내 자신의 노가다;
        """

    def loss_nl(self):
        patchS_stack = list()
        patchA_stack = list()
        patchB_stack = list()
        mixedS = list()
        mixedA = list()
        mixedB = list()

        index_map = np.zeros((2, self.height, self.width)).astype(np.int32)
        index_map[0] = np.arange(self.height)[:, np.newaxis] + np.zeros(self.width).astype(np.int32)
        index_map[1] = np.zeros(self.height).astype(np.int32)[:, np.newaxis] + np.arange(self.width)

        for i in range(5):
            index_map_cluster = index_map[:, self.kmeans_labels == i]
            source_cluster = self.source[index_map_cluster[0], index_map_cluster[1]].cpu()
            paramA_cluster = self.paramA[index_map_cluster[0], index_map_cluster[1]]
            paramB_cluster = self.paramB[index_map_cluster[0], index_map_cluster[1]]

            nbrs = NearestNeighbors(n_neighbors=9, n_jobs=1).fit(source_cluster)
            indices = nbrs.kneighbors(source_cluster, return_distance=False)
            source_cluster = source_cluster.cuda()
            patchS_stack.append(source_cluster[indices[:, 1:].reshape(-1)].reshape(-1, 8, self.channel))
            patchA_stack.append(paramA_cluster[indices[:, 1:].reshape(-1)].reshape(-1, 8, self.channel))
            patchB_stack.append(paramB_cluster[indices[:, 1:].reshape(-1)].reshape(-1, 8, self.channel))
            mixedS.append(source_cluster.unsqueeze(1))
            mixedA.append(paramA_cluster.unsqueeze(1))
            mixedB.append(paramB_cluster.unsqueeze(1))

        patchS_stack = torch.cat(patchS_stack)
        patchA_stack = torch.cat(patchA_stack)
        patchB_stack = torch.cat(patchB_stack)
        mixedS = torch.cat(mixedS)
        mixedA = torch.cat(mixedA)
        mixedB = torch.cat(mixedB)

        mixedT = mixedA * mixedS + mixedB
        patchT_stack = patchA_stack * patchS_stack + patchB_stack
        patchSD = torch.norm(mixedS - patchS_stack, 2, 2).exp()
        wgt = patchSD / patchSD.sum(1, keepdim=True)
        term1 = torch.pow(mixedT - patchT_stack, 2).sum(2)
        loss_nl = torch.sum(wgt * term1, 1).mean()

        return loss_nl

    def train(self, total_iter=250):
        optimizer = optim.Adam([self.paramA, self.paramB], lr=0.1, weight_decay=0)
        hyper_l = 0.125
        hyper_nl = 2.0
        for iter in range(total_iter):
            optimizer.zero_grad()

            loss_d = self.loss_d()
            loss_l = self.loss_l()
            loss_nl = self.loss_nl()
            loss = loss_d + hyper_l * loss_l + hyper_nl * loss_nl

            print("Loss_d: {0:.4f}, Loss_l: {1:.4f}, loss_nl: {2:.4f}".format(loss_d.data, loss_l.data, loss_nl.data))
            if (iter + 1) % 10 == 0:
                print("Iteration:", str(iter + 1) + "/" + str(total_iter), "Loss: {0:.4f}".format(loss.data))
            loss.backward()
            optimizer.step()

rgbOrigS = transforms.ToTensor()(origS)
labOrigS = transforms.ToTensor()(laborigS)

os.makedirs("results", exist_ok=True)
print("here")
lct = LocalColorTransfer(labS, labG, feat5S_norm, feat5G_norm, kmeans_ratio=1)
from guided_filter_pytorch.guided_filter import FastGuidedFilter
a_upsampled = FastGuidedFilter(1, eps=1e-08)(lct.source.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                             lct.paramA.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                             labOrigS.unsqueeze(0)).squeeze()
b_upsampled = FastGuidedFilter(1, eps=1e-08)(lct.source.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                             lct.paramB.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                             labOrigS.unsqueeze(0)).squeeze()

img5S = a_upsampled * (255*labOrigS) + b_upsampled
img5S = img5S.data.numpy().transpose(LEFT_SHIFT)
img5S = np.clip(img5S, 0, 255)
print(img5S.shape)
img5S = cv2.cvtColor(img5S.astype(np.uint8), cv2.COLOR_LAB2RGB)
cv2.imwrite('results/img5S.png', img5S)
# img5S = img5S.data.numpy().transpose(LEFT_SHIFT).astype(np.uint8)
print(img5S)

# img5S = color.lab2rgb(img5S.data.numpy().transpose(LEFT_SHIFT).astype(np.float64))
# imshow(img5S)

# img5S = torch.from_numpy(img5S.transpose(RIGHT_SHIFT)).float()
save = torch.from_numpy(imgG).float()
print(save)
utils.save_image(save, 'results/img5G.png')
lct.visualize()

lct.train()
lct.visualize()

# labOrigS = torch.from_numpy(color.rgb2lab(np.array(origS)).transpose(RIGHT_SHIFT)).float()
a_upsampled = FastGuidedFilter(1, eps=1e-08)(lct.source.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                             lct.paramA.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                             labOrigS.unsqueeze(0)).squeeze()
b_upsampled = FastGuidedFilter(1, eps=1e-08)(lct.source.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                             lct.paramB.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                             labOrigS.unsqueeze(0)).squeeze()

tsshow(lct.paramA.data.permute(RIGHT_SHIFT).cpu())
tsshow(a_upsampled.data)
tsshow(lct.paramB.data.permute(RIGHT_SHIFT).cpu())
tsshow(b_upsampled.data)

img5S = a_upsampled * (255*labOrigS) + b_upsampled
img5S = img5S.data.numpy().transpose(LEFT_SHIFT)
img5S = np.clip(img5S, 0, 255)
print(img5S.shape)
img5S = cv2.cvtColor(img5S.astype(np.uint8), cv2.COLOR_LAB2RGB)
cv2.imwrite('results/img5S.png', img5S)

transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img5S/255)
img5S = img5S.unsqueeze(0)

feat4S = get_feature(img5S, FEATURE_IDS[3])
feat4R = get_feature(imgR, FEATURE_IDS[3])
feat4S_norm = normalize(feat4S)
feat4R_norm = normalize(feat4R)

map4SR = PatchMatch(feat4S_norm, feat4R_norm) #S -> R
map4RS = PatchMatch(feat4R_norm, feat4S_norm) #R -> S
map4SR.solve()
print()
map4RS.solve()

imgS_resized = resize_img(origS, feat4S.shape[:2])
imgR_resized = resize_img(origR, feat4R.shape[:2])

imgG = bds_vote(imgR_resized.numpy(), map4SR.nnf, map4RS.nnf)
feat4G = bds_vote(feat4R.transpose(RIGHT_SHIFT), map4SR.nnf, map4RS.nnf).transpose(LEFT_SHIFT)
feat4G_norm = normalize(feat4G)

labS = color.rgb2lab(imgS_resized.numpy().transpose(LEFT_SHIFT))
labG = color.rgb2lab(imgG.transpose(LEFT_SHIFT))

lct = LocalColorTransfer(labS, labG, feat4S_norm, feat4G_norm, L=1, kmeans_ratio=2)
save = torch.from_numpy(imgG).float()
utils.save_image(save, 'results/img4G.png')
lct.train()

a_upsampled = FastGuidedFilter(1, eps=1e-08)(lct.source.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                             lct.paramA.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                             labOrigS.unsqueeze(0)).squeeze()
b_upsampled = FastGuidedFilter(1, eps=1e-08)(lct.source.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                             lct.paramB.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                             labOrigS.unsqueeze(0)).squeeze()

img4S = a_upsampled * labOrigS + b_upsampled
img4S = img4S.data.numpy().transpose(LEFT_SHIFT)
# imshow(img4S)

img4S = torch.from_numpy(img4S.transpose(RIGHT_SHIFT)).float()
utils.save_image(img4S, 'results/img4S.png')
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img4S)
img4S = img4S.unsqueeze(0)

feat3S = get_feature(img4S, FEATURE_IDS[2])
feat3R = get_feature(imgR, FEATURE_IDS[2])
feat3S_norm = normalize(feat3S)
feat3R_norm = normalize(feat3R)

map3SR = PatchMatch(feat3S_norm, feat3R_norm) #S -> R
map3RS = PatchMatch(feat3R_norm, feat3S_norm) #R -> S
map3SR.solve()
print()
map3RS.solve()

imgS_resized = resize_img(origS, feat3S.shape[:2])
imgR_resized = resize_img(origR, feat3R.shape[:2])

imgG = bds_vote(imgR_resized.numpy(), map3SR.nnf, map3RS.nnf)
feat3G = bds_vote(feat3R.transpose(RIGHT_SHIFT), map3SR.nnf, map3RS.nnf).transpose(LEFT_SHIFT)
feat3G_norm = normalize(feat3G)

labS = color.rgb2lab(imgS_resized.numpy().transpose(LEFT_SHIFT))
labG = color.rgb2lab(imgG.transpose(LEFT_SHIFT))

lct = LocalColorTransfer(labS, labG, feat3S_norm, feat3G_norm, L=2, kmeans_ratio=4)
save = torch.from_numpy(imgG).float()
utils.save_image(save, 'results/img3G.png')
lct.train()

a_upsampled = FastGuidedFilter(1, eps=1e-08)(lct.source.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                             lct.paramA.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                             labOrigS.unsqueeze(0)).squeeze()
b_upsampled = FastGuidedFilter(1, eps=1e-08)(lct.source.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                             lct.paramB.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                             labOrigS.unsqueeze(0)).squeeze()

img3S = a_upsampled * labOrigS + b_upsampled
img3S = img3S.data.numpy().transpose(LEFT_SHIFT)
# imshow(img3S)

img3S = torch.from_numpy(img3S.transpose(RIGHT_SHIFT)).float()
utils.save_image(img3S, 'results/img3S.png')
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img3S)
img3S = img3S.unsqueeze(0)

feat2S = get_feature(img3S, FEATURE_IDS[1])
feat2R = get_feature(imgR, FEATURE_IDS[1])
feat2S_norm = normalize(feat2S)
feat2R_norm = normalize(feat2R)

map2SR = PatchMatch(feat2S_norm, feat2R_norm) #S -> R
map2RS = PatchMatch(feat2R_norm, feat2S_norm) #R -> S
map2SR.solve()
print()
map2RS.solve()

imgS_resized = resize_img(origS, feat2S.shape[:2])
imgR_resized = resize_img(origR, feat2R.shape[:2])

imgG = bds_vote(imgR_resized.numpy(), map2SR.nnf, map2RS.nnf)
feat2G = bds_vote(feat2R.transpose(RIGHT_SHIFT), map2SR.nnf, map2RS.nnf).transpose(LEFT_SHIFT)
feat2G_norm = normalize(feat2G)

labS = color.rgb2lab(imgS_resized.numpy().transpose(LEFT_SHIFT))
labG = color.rgb2lab(imgG.transpose(LEFT_SHIFT))

lct = LocalColorTransfer(imgS_resized.numpy().transpose(LEFT_SHIFT), imgG.transpose(LEFT_SHIFT), feat2S_norm, feat2G_norm, L=3, kmeans_ratio=8)
save = torch.from_numpy(imgG).float()
utils.save_image(save, 'results/img2G.png')
lct.train()

a_upsampled = FastGuidedFilter(1, eps=1e-08)(lct.source.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                             lct.paramA.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                             labOrigS.unsqueeze(0)).squeeze()
b_upsampled = FastGuidedFilter(1, eps=1e-08)(lct.source.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                             lct.paramB.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                             labOrigS.unsqueeze(0)).squeeze()

img2S = a_upsampled * labOrigS + b_upsampled
img2S = img2S.data.numpy().transpose(LEFT_SHIFT)
# imshow(img2S)

img2S = torch.from_numpy(img2S.transpose(RIGHT_SHIFT)).float()
utils.save_image(img2S, 'results/img2S.png')
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img2S)
img2S = img2S.unsqueeze(0)

feat1S = get_feature(img2S, FEATURE_IDS[0])
feat1R = get_feature(imgR, FEATURE_IDS[0])
feat1S_norm = normalize(feat1S)
feat1R_norm = normalize(feat1R)

map1SR = PatchMatch(feat1S_norm, feat1R_norm) #S -> R
map1RS = PatchMatch(feat1R_norm, feat1S_norm) #R -> S
map1SR.solve()
print()
map1RS.solve()

imgS_resized = resize_img(origS, feat1S.shape[:2])
imgR_resized = resize_img(origR, feat1R.shape[:2])

imgG = bds_vote(imgR_resized.numpy(), map1SR.nnf, map1RS.nnf)
feat1G = bds_vote(feat1R.transpose(RIGHT_SHIFT), map1SR.nnf, map1RS.nnf).transpose(LEFT_SHIFT)
feat1G_norm = normalize(feat1G)

labS = color.rgb2lab(imgS_resized.numpy().transpose(LEFT_SHIFT))
labG = color.rgb2lab(imgG.transpose(LEFT_SHIFT))

lct = LocalColorTransfer(imgS_resized.numpy().transpose(LEFT_SHIFT), imgG.transpose(LEFT_SHIFT), feat1S_norm, feat1G_norm, L=4, kmeans_ratio=16)
save = torch.from_numpy(imgG).float()
utils.save_image(save, 'results/img1G.png')
lct.train()

a_upsampled = FastGuidedFilter(1, eps=1e-08)(lct.source.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                             lct.paramA.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                             labOrigS.unsqueeze(0)).squeeze()
b_upsampled = FastGuidedFilter(1, eps=1e-08)(lct.source.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                             lct.paramB.permute(RIGHT_SHIFT).unsqueeze(0).cpu(),
                                             labOrigS.unsqueeze(0)).squeeze()

img1S = a_upsampled * labOrigS + b_upsampled
img1S = img1S.data.numpy().transpose(LEFT_SHIFT)
# imshow(img1S)

img1S = torch.from_numpy(img1S.transpose(RIGHT_SHIFT)).float()
utils.save_image(img1S, 'results/img1S.png')