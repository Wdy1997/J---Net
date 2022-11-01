import torch
import torch.nn as nn
import warnings
import torch.nn.functional as F
from lib.Res2Net_v1b import res2net50_v1b_26w_4s
from  lib.CFP import CFPModule
from lib.mixpool import MixPool
import cv2
import numpy as np

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class M_C(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(M_C, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2

        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3


        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)

        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)
        return x


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Multiscale(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, size =tuple([88,44,22,11]),channel=256):
        super().__init__()
        embedding_dim = [channel,channel*2,channel*3,channel*4]
        kernel=[3,7,11,15]
        self.out= []
        for i in size:
            self.out.append([i,i])
        self.BN = nn.BatchNorm2d(channel)
        self.blk = []
        for i in range(len(self.out)):
            self.blk.append(nn.Sequential(nn.Conv2d(embedding_dim[i], embedding_dim[i], kernel_size=kernel[i], padding=int((kernel[i]-1)/2), bias=False),nn.ReLU()))
        self.conv = nn.Sequential(nn.Conv2d(embedding_dim[-1], embedding_dim[0], 3, padding=1, bias=False))


    def forward(self, x):
        initial = x
        out = []

        for i in range(len(self.out)):
            self.blk[i] = self.blk[i].cuda()
            x = resize(x, size=self.out[i], mode='bilinear', align_corners=False)
            out.append(x)

        out_0 = self.blk[0](out[0])
        out_0 = resize(out_0, size=out[1].shape[2:], mode='bilinear', align_corners=False)

        out_1 = torch.cat([out_0, out[1]], dim=1)
        out_1 = self.blk[1](out_1)
        out_1 = resize(out_1, size=out[2].shape[2:], mode='bilinear', align_corners=False)

        out_2 = torch.cat([out_1, out[2]], dim=1)
        out_2 = self.blk[2](out_2)
        out_2 = resize(out_2, size=out[3].shape[2:], mode='bilinear', align_corners=False)

        out_3 = torch.cat([out_2, out[3]], dim=1)
        out_3 = self.blk[3](out_3)
        out_3 = resize(out_3, size=initial.shape[2:], mode='bilinear', align_corners=False)

        x = self.conv(out_3)
        x = x + initial
        return x



class J_Net(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32):
        super(J_Net, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        # ---- Receptive Field Block like module ----
        self.rfb2_1 = nn.Conv2d(512, 32,kernel_size=3,padding=1)

        self.rfb3_1 = nn.Sequential(nn.Conv2d(1024, 32,kernel_size=3,padding=1))

        self.rfb4_1 = nn.Sequential(nn.Conv2d(2048, 32,kernel_size=3,padding=1))


        # embedding_dim = 256
        self.Multiscale1_1 = Multiscale(channel=32)
        self.Multiscale2_1 = Multiscale(channel=32)
        self.Multiscale3_1 = Multiscale(channel=32)
        self.Multiscale4_1 = Multiscale(channel=32)

        # ---- Partial Decoder ----
        self.agg1 = M_C(channel)

        # ---- reverse attention branch 4 ----
        self.ra4_conv1 = BasicConv2d(2048, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, 1, kernel_size=1)




        # ---- reverse attention branch 3 ----
        self.ra3_conv1 = BasicConv2d(1024, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

        # ---- reverse attention branch 2 ----
        self.ra2_conv1 = BasicConv2d(512, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)


        #------
        self.CFP_1 = CFPModule(2048, d=8)
        self.CFP_2 = CFPModule(1024, d=8)
        self.CFP_3 = CFPModule(512, d=8)

        self.p4=MixPool(2048,2048)
        self.p3=MixPool(1024,1024)
        self.p2=MixPool(512,512)

    #def dotProduct(self,seg,cls):
        #B, N, H, W = seg.size()
       # seg = seg.view(B, N, H * W)
        #final = torch.einsum("ijk,ij->ijk", [seg, cls])
        #final = final.view(B, N, H, W)
      #  return final

    def forward(self, x):

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)      # bs, 64, 88, 88
        # ---- low-level features ----
        x1 = self.resnet.layer1(x)      # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44
        x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)     # bs, 2048, 11, 11
        x2_rfb = self.rfb2_1(x2)        # channel -> 32

        x3_rfb = self.rfb3_1(x3)        # channel -> 32

        x4_rfb = self.rfb4_1(x4)        # channel -> 32

        x2_rfb = self.Multiscale1_1(x2_rfb)
        x3_rfb = self.Multiscale2_1(x3_rfb)
        x4_rfb = self.Multiscale3_1(x4_rfb)

        ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb)

        lateral_map_5 = F.interpolate(ra5_feat, scale_factor=8, mode='bilinear')    # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
        # ---- reverse attention branch_4 ----
        crop_4 = F.interpolate(ra5_feat, scale_factor=0.25, mode='bilinear')
        x = -1 * (torch.sigmoid(crop_4)) + 1
        x = x.expand(-1, x4.shape[1], -1, -1).mul(x4)
        x = self.ra4_conv1(x)
        x = F.relu(self.ra4_conv2(x))
        x = F.relu(self.ra4_conv3(x))
        ra4_feat = self.ra4_conv5(x)
        x = ra4_feat +crop_4

        x44_Copy=torch.rand_like(x4)
        # print("4<<<<<<<<<<<<<<<<<<<")
        if (bool(int(abs(x.sum()))!=int(abs(crop_4.sum())+abs(ra4_feat.sum())))):
            # print("4>>>>>>>>>>>>>>>>")
            x = torch.sigmoid(crop_4)
            for j in range(x4.shape[0]):
                for i in range(x4.shape[1]):
                    x44_Copy[j][i] = -x4[j][i] + x4[j][i].max()
            x = x.expand(-1, x44_Copy.shape[1], -1, -1).mul(x44_Copy)
            x = self.ra4_conv1(x)
            x = F.relu(self.ra4_conv2(x))
            x = F.relu(self.ra4_conv3(x))
            ra4_feat = self.ra4_conv5(x)
            x = ra4_feat + crop_4
            print("==", bool(int(abs(x.sum()))==int(abs(crop_4.sum())+abs(ra4_feat.sum()))))

        ra4_feat = F.interpolate(ra4_feat, scale_factor=32, mode='bilinear')
        lateral_map_4 = F.interpolate(x, scale_factor=32, mode='bilinear')  # NOTES: Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)
        # ---- reverse attention branch_3 ----
        crop_3 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1 * (torch.sigmoid(crop_3)) + 1
        x = x.expand(-1, x3.shape[1], -1, -1).mul(x3)
        x = self.ra3_conv1(x)
        x = F.relu(self.ra3_conv2(x))
        x = F.relu(self.ra3_conv3(x))
        ra3_feat = self.ra3_conv4(x)
        x = ra3_feat + crop_3

        x33_Copy=torch.rand_like(x3)
        # print("3<<<<<<<<<<<<<<<<<<<")
        if (bool(int(abs(x.sum()))!=int(abs(crop_3.sum())+abs(ra3_feat.sum())))):
            # print("3>>>>>>>>>>>>>>>>")
            x = torch.sigmoid(crop_3)
            for j in range(x3.shape[0]):
                for i in range(x3.shape[1]):
                    x33_Copy[j][i] = -x3[j][i] + x3[j][i].max()
            x = x.expand(-1, x33_Copy.shape[1], -1, -1).mul(x33_Copy)
            x = self.ra3_conv1(x)
            x = F.relu(self.ra3_conv2(x))
            x = F.relu(self.ra3_conv3(x))
            ra3_feat = self.ra3_conv4(x)
            x = ra3_feat + crop_3
            print("==", bool(int(abs(x.sum())) == int(abs(crop_3.sum()) + abs(ra3_feat.sum()))))

        ra3_feat = F.interpolate(ra3_feat, scale_factor=16, mode='bilinear')
        lateral_map_3 = F.interpolate(x, scale_factor=16, mode='bilinear')  # NOTES: Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)
        # ---- reverse attention branch_2 ----
        crop_2 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1 * (torch.sigmoid(crop_2)) + 1
        x = x.expand(-1, x2.shape[1], -1, -1).mul(x2)
        x = self.ra2_conv1(x)
        x = F.relu(self.ra2_conv2(x))
        x = F.relu(self.ra2_conv3(x))
        ra2_feat = self.ra2_conv4(x)
        x = ra2_feat + crop_2

        # print("2<<<<<<<<<<<<<<<<<<<")
        x22_Copy = torch.rand_like(x2)
        if (bool(int(abs(x.sum()))!=int(abs(crop_2.sum())+abs(ra2_feat.sum())))):
            # print("2>>>>>>>>>>>>>>>>")
            x = torch.sigmoid(crop_2)
            for j in range(x2.shape[0]):
                for i in range(x2.shape[1]):
                    x22_Copy[j][i] = -x2[j][i] + x2[j][i].max()
            x = x.expand(-1, x22_Copy.shape[1], -1, -1).mul(x22_Copy)
            x = self.ra2_conv1(x)
            x = F.relu(self.ra2_conv2(x))
            x = F.relu(self.ra2_conv3(x))
            ra2_feat = self.ra2_conv4(x)
            x = ra2_feat + crop_2
            print("==", bool(int(abs(x.sum())) == int(abs(crop_2.sum()) + abs(ra2_feat.sum()))))

        ra2_feat=F.interpolate(ra2_feat, scale_factor=8, mode='bilinear')
        lateral_map_2 = F.interpolate(x, scale_factor=8, mode='bilinear')   # NOTES: Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2,ra4_feat,ra3_feat,ra2_feat


if __name__ == '__main__':
    input=torch.rand(4,3,352,352).cuda()
    net = J_Net().cuda()
    lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2,ra4_feat,ra3_feat,ra2_feat=net(input)
    print(lateral_map_5.shape)
    print(ra4_feat.shape)
    print(ra3_feat.shape)
    print(ra2_feat.shape)


