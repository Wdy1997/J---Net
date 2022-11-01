import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from lib.J_Net_Res2Net_J_RA import J_Net                                                               ######修改位置1##########
# from lib.PraNet_swintransform import PraNet
from utils.dataloader import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='snapshots/J_RA/Best_PraNet.pth')                   ######修改位置2##########


for _data_name in ["Kvasir","ETIS-LaribPolypDB","CVC-300","CVC-ClinicDB","CVC-ColonDB"]:
    data_path = r"C:\Users\Ad'min\DeepLearning\torch\deep-learning-for-image-processing-master\pytorch_segmentation\data\息肉1\test/{}/".format(_data_name)
    save_path_1 = './results/PraNet/{}/'.format(_data_name)
    save_path_2 = r'./results/PraNet/NEW/Res2Net\多尺度判断传播_凯哥数据集\J_RA/{}/'.format(_data_name)     ######修改位置3##########
    save_path_3 = './results/PraNet/边界/{}/'.format(_data_name)
    save_path = r'{}'.format(save_path_2+_data_name)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path_3+_data_name, exist_ok=True)
    opt = parser.parse_args()
    model = J_Net()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path_1, exist_ok=True)
    os.makedirs(save_path_2, exist_ok=True)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/1st_manual/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    # for name,param in model.named_parameters():
    #     print(name)


    k=0
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        k=(k+1)
        res5, res4, res3, res2,ra4_feat,ra3_feat,ra2_feat = model(image)
        for n,i in enumerate(model(image)):
            # print(n)
            # res = i
            # res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            # res = res.sigmoid().data.cpu().numpy().squeeze()
            # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            # plt.imshow(res, cmap=plt.get_cmap('gray'))
            # plt.imsave(save_path_1 +f'_{k}_'+f'{n%4}.png', res, cmap="gray")

            if(n==3):
                res = i
                res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                plt.imshow(res, cmap=plt.get_cmap('gray'))
                plt.imsave(save_path_2 + name, res, cmap="gray")
            if(n==6):
                res = i
                res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                plt.imshow(res, cmap=plt.get_cmap('gray'))
                plt.imsave(save_path_3 + name, res, cmap="gray")
