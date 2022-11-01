import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F


class GaussianBlur(nn.Module):
    def __init__(self):
        super(GaussianBlur, self).__init__()
        kernel = [[1 / 49, 1 / 49, 1 / 49,1 / 49, 1 / 49, 1 / 49, 1 / 49],
                  [1 / 49, 1 / 49, 1 / 49,1 / 49, 1 / 49, 1 / 49, 1 / 49],
                  [1 / 49, 1 / 49, 1 / 49,1 / 49, 1 / 49, 1 / 49, 1 / 49],
                  [1 / 49, 1 / 49, 1 / 49,1 / 49, 1 / 49, 1 / 49, 1 / 49],
                  [1 / 49, 1 / 49, 1 / 49,1 / 49, 1 / 49, 1 / 49, 1 / 49],
                  [1 / 49, 1 / 49, 1 / 49,1 / 49, 1 / 49, 1 / 49, 1 / 49],
                  [1 / 49, 1 / 49, 1 / 49,1 / 49, 1 / 49, 1 / 49, 1 / 49]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0).cuda()
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x1 = x[:, 0, :, :]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=3)
        return x1


class dilate(nn.Module):
    def __init__(self):
        super(dilate, self).__init__()
        kernel = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0).cuda()
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x1 = x[:, 0, :, :]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=1)
        return x1


class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        kerne_x = [[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]]
        kerne_y = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kerne_x = torch.FloatTensor(kerne_x).unsqueeze(0).unsqueeze(0).cuda()
        kerne_y = torch.FloatTensor(kerne_y).unsqueeze(0).unsqueeze(0).cuda()
        self.weight_x = nn.Parameter(data=kerne_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kerne_y, requires_grad=False)

    def forward(self, x):
        x1 = x[:, 0, :, :]
        x1_x = F.conv2d(x1.unsqueeze(1), self.weight_x, padding=1)
        x1_y = F.conv2d(x1.unsqueeze(1), self.weight_y, padding=1)
        x1 = torch.sqrt(x1_x ** 2 + x1_y ** 2)
        return x1


def threshold(img):
    img_copy=torch.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_copy[i][j] = (img[i][j] - img[i][j].min()) / (img[i][j].max() - img[i][j].min())
    img_copy[img_copy <= 0.1] = 0
    img_copy[img_copy > 0.1] = 1
    return img_copy

gaussianBlur=GaussianBlur()
Sobel=Sobel()
dilate=dilate()
net=nn.Sequential(gaussianBlur,Sobel,dilate).cuda()
