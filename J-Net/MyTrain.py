import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.J_Net_Res2Net_J_RA import J_Net
# from lib.PraNet_swintransform import PraNet
from utils.dataloader import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import miou
import torch.nn.functional as F
import numpy as np
import Image_processing as img_pro
import miou

def structure_loss(pred,ra_feat,mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    bd_iou=0
    if(ra_feat is not None):
        img_Canny = img_pro.net(mask)
        img_Canny[img_Canny > 0.5]=1
        img_Canny[img_Canny <= 0.5] = 0
        thresh4 = img_pro.threshold(ra_feat)
        Miou________, DIce________ = miou.meanIou(thresh4,img_Canny,classNum=2)
        bd_iou = 1-5*(Miou________+DIce________)
    return (wbce + wiou + bd_iou).mean()


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
            lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, ra4_feat, ra3_feat, ra2_feat = model(images)

            loss5 = structure_loss(lateral_map_5,None,gts)
            loss4 = structure_loss(lateral_map_4,ra4_feat,gts)
            loss3 = structure_loss(lateral_map_3,ra3_feat,gts)
            loss2 = structure_loss(lateral_map_2,ra2_feat,gts)
            loss = loss2 + loss3 + loss4 + loss5   # TODO: try different weights for loss
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record2.update(loss2.data, opt.batchsize)
                loss_record3.update(loss3.data, opt.batchsize)
                loss_record4.update(loss4.data, opt.batchsize)
                loss_record5.update(loss5.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}, lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record2.show(), loss_record3.show(), loss_record4.show(), loss_record5.show()))
    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), save_path + 'PraNet-%d.pth' % epoch)
        print('[Saving Snapshot:]', save_path + 'PraNet-%d.pth'% epoch)




def validate(val_loader,model):
    model.eval()
    val_loss=[]
    MIOU = []
    DICE = []
    global record_iou
    with torch.no_grad():
        size_rates = [0.75, 1, 1.25]
        loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        for i,pack in enumerate(val_loader, start=1):
            for rate in size_rates:
                images, gts = pack
                images = Variable(images).cuda()
                gts = Variable(gts).cuda()
                # ---- rescale ----
                trainsize = 352
                if rate != 1:
                    images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                # ---- forward ----
                lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, ra4_feat, ra3_feat, ra2_feat = model(images)
                loss5 = structure_loss(lateral_map_5, None, gts)
                loss4 = structure_loss(lateral_map_4, ra4_feat, gts)
                loss3 = structure_loss(lateral_map_3, ra3_feat, gts)
                loss2 = structure_loss(lateral_map_2, ra2_feat, gts)
                # ---- loss function ----
                loss = loss2 + loss3 + loss4 + loss5  # TODO: try different weights for loss
                val_loss.append(loss.item())

                lateral_map_2 = F.upsample(lateral_map_2, size=gts.shape[2:], mode='bilinear', align_corners=False)

                preimage=lateral_map_2.sigmoid()
                preimage_Copy=torch.zeros_like(preimage)
                for j in range(preimage.shape[0]):
                    for i in range(preimage.shape[1]):
                        preimage_Copy[j][i] = (preimage[j][i] - preimage[j][i].min()) / (preimage[j][i].max() - preimage[j][i].min() + 1e-8)
                preimage_Copy[preimage_Copy >= 0.5] = 1
                preimage_Copy[preimage_Copy < 0.5] = 0
                gts[gts >= 0.5] = 1
                gts[gts < 0.5] = 0
                MIOU_,DICE_=miou.meanIou(preimage_Copy,gts,classNum=2)
                MIOU.append(MIOU_)
                DICE.append(DICE_)
    print("mean_loss: {:.4f}, mean_iou: {:0.4f}, mean_dice: {:0.4f}".format(np.mean(val_loss),np.mean(MIOU),np.mean(DICE)))
    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if record_iou < (np.mean(MIOU)):
        record_iou = np.mean(MIOU)
        torch.save(model.state_dict(), save_path + 'Best_PraNet.pth')
        print('[Saving Snapshot:]', save_path + 'Best_PraNet_%d.pth'% epoch)
    return np.mean(val_loss), np.mean(MIOU), np.mean(DICE)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int,
                        default=40, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int,
                        default=8, help='training batch size')
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
                        default='J_RA')
    parser.add_argument('--resume', type=str,default=r"")
    # parser.add_argument('--resume', type=str,default=r"C:\Users\Ad'min\DeepLearning\torch\deep-learning-for-image-processing-master——1\pytorch_segmentation\PRANET\snapshots\PraNet_Res2Net_copy\M+J_RA+bd_loss\Best_PraNet.pth")
    opt = parser.parse_args()

    # ---- build models ----
    model = J_Net().cuda()

####################################################载入预训练权重##################################################################
    if opt.resume != "":
        checkpoint = torch.load(opt.resume, map_location='cpu')
        model.load_state_dict(checkpoint,strict=False)
#################################################################################################################################

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
    record_iou = 0
    txt_file=os.open("data_{}.txt".format(opt.train_save) ,os.O_RDWR|os.O_CREAT)
    f = open(txt_file, "w")
    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch)
        val_loss_data,MIOU_data,DICE_data=validate(eval_loader,model)
        Str = "mean_loss: {:.4f}, mean_iou: {:0.4f}, mean_dice: {:0.4f}\n".format(np.mean(val_loss_data),np.mean(MIOU_data),np.mean(DICE_data))
        f.write(Str + "\n\n")
    f.close()