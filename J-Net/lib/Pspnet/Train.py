import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
import PSPNET
# from lib.PraNet_swintransform import PraNet
from utils.dataloader import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import miou
import torch.nn.functional as F
import numpy as np
import Image_processing as img_pro
import miou

def structure_loss(pred,mask):

    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou ).mean()

def train(train_loader, model, optimizer, epoch):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]
    loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            #canny_image = Variable(canny_image).cuda()
            # ---- rescale ----
            trainsize = 352
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            output = model(images)
            loss = structure_loss(output,gts)
            # ---- backward ----
            loss = loss.requires_grad_()
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record5.update(loss.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}],lateral: {:0.4f} '.format(datetime.now(), epoch, opt.epoch, i, total_step,loss_record5.show()))
    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), save_path + 'PraNet-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'PraNet-%d.pth'% epoch)


# def validate(val_loader,model):
#     model.eval()
#     val_loss=[]
#     MIOU = []
#     DICE = []
#     with torch.no_grad():
#         size_rates = [0.75, 1, 1.25]
#         loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
#         for i,pack in enumerate(val_loader, start=1):
#             for rate in size_rates:
#                 images, gts = pack
#                 images = Variable(images).cuda()
#                 gts = Variable(gts).cuda()
#                 # ---- rescale ----
#                 trainsize = 352
#                 if rate != 1:
#                     images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
#                     gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
#                     img_Canny = img_pro.net(gts)
#                 # ---- forward ----
#                 lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2,ra4_feat,ra3_feat,ra2_feat = model(images)
#                 thresh2 = img_pro.threshold(ra2_feat)
#                 thresh3 = img_pro.threshold(ra3_feat)
#                 thresh4 = img_pro.threshold(ra4_feat)
#                 # ---- loss function ----
#                 loss8 = structure_loss(thresh4, img_Canny)
#                 loss7 = structure_loss(thresh3, img_Canny)
#                 loss6 = structure_loss(thresh2,img_Canny)
#                 loss5 = structure_loss(lateral_map_5, gts)
#                 loss4 = structure_loss(lateral_map_4, gts)
#                 loss3 = structure_loss(lateral_map_3, gts)
#                 loss2 = structure_loss(lateral_map_2, gts)
#                 loss = loss2 + loss3 + loss4 + loss5 + loss6+ loss7+ loss8  # TODO: try different weights for loss
#                 val_loss.append(loss.item())
#                 preimage=lateral_map_2.sigmoid()
#                 preimage_Copy=torch.zeros_like(preimage)
#                 for j in range(preimage.shape[0]):
#                     for i in range(preimage.shape[1]):
#                         preimage_Copy[j][i] = (preimage[j][i] - preimage[j][i].min()) / (preimage[j][i].max() - preimage[j][i].min() + 1e-8)
#                 preimage_Copy[preimage_Copy >= 0.5] = 1
#                 preimage_Copy[preimage_Copy < 0.5] = 0
#                 gts[gts >= 0.5] = 1
#                 gts[gts < 0.5] = 1
#                 MIOU_,DICE_=miou.meanIou(preimage_Copy,gts,classNum=2)
#                 MIOU.append(MIOU_)
#                 DICE.append(DICE_)
#     print("mean_loss: {:.4f}, mean_iou: {:0.4f}, mean_dice: {:0.4f}".format(np.mean(val_loss),np.mean(MIOU),np.mean(DICE)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=40, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=1, help='training batch size')
    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.9, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.98, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=4, help='every n epochs decay learning rate')
    parser.add_argument('--train_path', type=str,
                        default=r"C:\Users\Ad'min\DeepLearning\torch\deep-learning-for-image-processing-master\pytorch_segmentation\data\息肉1\training", help='path to train dataset')
    parser.add_argument('--eval_path', type=str,
                        default=r"C:\Users\Ad'min\DeepLearning\torch\deep-learning-for-image-processing-master\pytorch_segmentation\data\息肉1\test\Kvasir",
                        help='path to train dataset')
    parser.add_argument('--train_save', type=str,
                        default='PSP-Net')
    # parser.add_argument('--resume', type=str,default=r"C:\Users\Ad'min\DeepLearning\torch\deep-learning-for-image-processing-master——1\pytorch_segmentation\PRANET\snapshots\PraNet_Res2Net_copy\PraNet-9.pth")
    opt = parser.parse_args()

    # ---- build models ----
    model = PSPNET.Pspnet(num_classes=2).cuda()
    best_loss=1000.0

#####################################################载入预训练权重##################################################################
    # if opt.resume:
    #     checkpoint = torch.load(opt.resume, map_location='cpu')
    #     model.load_state_dict(checkpoint,strict=False)
##################################################################################################################################

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/1st_manual/'.format(opt.train_path)
    eval_image_root = '{}/images/'.format(opt.eval_path)
    eval_gt_root = '{}/1st_manual/'.format(opt.eval_path)

    print(image_root)
    print(gt_root)
    print(eval_image_root)
    print(eval_gt_root)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    eval_loader=get_loader(eval_image_root, eval_gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)
    print("#"*20, "Start Training", "#"*20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch)
        # validate(eval_loader,model)