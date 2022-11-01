import os
import cv2
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
import math
from train_utils.train_and_eval import train_one_epoch, evaluate, create_lr_scheduler
import train_utils.distributed_utils as utils
__all__ = ['SegmentationMetric']

import PIL.Image as Image
"""
confusionMetric  # 注意：此处横着代表预测值，竖着代表真实值，与之前介绍的相反
P\L     P    N
P      TP    FP
N      FN    TN
"""



import numpy as np
import torch





class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()

        meanAcc = np.mean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0

        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)
        # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        return IoU


    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        # hist = np.bincount(
        #     self.numClass * imgLabel[mask].astype(int) +
        #     imgPredict[mask], minlength=self.numClass ** 2).reshape(self.numClass, self.numClass)
        # return hist

        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

    def dice_coeff(self,pred, target):
        smooth=1.
        num=pred.size(0.0)

        m1=pred.view(num,-1)
        m2=target.view(num,-1)
        intersection=(m1*m2).sum()
        return (2*intersection+smooth)/(m1.sum()+m2.sum()+smooth)

def meanIou(input, target, classNum):
    '''
    :param input: [b,h,w]
    :param target: [b,h,w]
    :param classNum: scalar
    :return:
    '''
    input=input.type(torch.LongTensor)
    target=target.type(torch.LongTensor)
    inputTmp = torch.zeros([input.shape[0], classNum, input.shape[2], input.shape[3]])  # 创建[b,c,h,w]大小的0矩阵
    targetTmp = torch.zeros([target.shape[0], classNum, target.shape[2], target.shape[3]])  # 同上
    inputOht = inputTmp.scatter_(index=input, dim=1, value=1)  # input作为索引，将0矩阵转换为onehot矩阵
    targetOht = targetTmp.scatter_(index=target, dim=1, value=1)  # 同上
    batchMious = []  # 为该batch中每张图像存储一个miou
    batchMdices = []
    mul = inputOht * targetOht  # 乘法计算后，其中1的个数为intersection
    for i in range(input.shape[0]):  # 遍历图像
        ious = []
        dices = []
        for j in range(classNum):  # 遍历类别，包括背景
            intersection = torch.sum(mul[i][j])
            union = torch.sum(inputOht[i][j]) + torch.sum(targetOht[i][j]) - intersection + 1e-6
            iou = intersection / union
            undice = torch.sum(inputOht[i][j]) + torch.sum(targetOht[i][j]) + 1e-6
            dice =2*intersection / undice
            ious.append(iou)
            dices.append(dice)
        miou = np.mean(ious)  # 计算该图像的miou
        mdice= np.mean(dices)  # 计算该图像的miou
        batchMious.append(miou)
        batchMdices.append(mdice)
    return np.mean(batchMious),np.mean(batchMdices)



def Iou(input, target, classNum):
    '''
    :param input: [b,h,w]
    :param target: [b,h,w]
    :param classNum: scalar
    :return:
    '''
    input=torch.LongTensor(input.reshape(1,input.shape[0],input.shape[1]))
    target=torch.LongTensor(target.reshape(1,target.shape[0],target.shape[1]))
    inputTmp = torch.zeros([input.shape[0], classNum, input.shape[1], input.shape[2]])  # 创建[b,c,h,w]大小的0矩阵
    targetTmp = torch.zeros([target.shape[0], classNum, target.shape[1], target.shape[2]])  # 同上
    input = input.unsqueeze(1)  # 将input维度扩充为[b,1,h,w]
    target = target.unsqueeze(1)  # 同上
    inputOht = inputTmp.scatter_(index=input, dim=1, value=1)  # input作为索引，将0矩阵转换为onehot矩阵
    targetOht = targetTmp.scatter_(index=target, dim=1, value=1)  # 同上
    batchMious = []  # 为该batch中每张图像存储一个miou
    batchMdices = []
    mul = inputOht * targetOht  # 乘法计算后，其中1的个数为intersection
    for i in range(input.shape[0]):  # 遍历图像
        ious = []
        dices = []
        for j in range(classNum):  # 遍历类别，包括背景
            intersection = torch.sum(mul[i][j])
            union = torch.sum(inputOht[i][j]) + torch.sum(targetOht[i][j]) - intersection + 1e-6
            iou = intersection / union
            undice = torch.sum(inputOht[i][j]) + torch.sum(targetOht[i][j]) + 1e-6
            dice =2*intersection / undice
            ious.append(iou)
            dices.append(dice)
        miou = np.mean(ious)  # 计算该图像的miou
        mdice= np.mean(dices)  # 计算该图像的miou
        batchMious.append(miou)
        batchMdices.append(mdice)
    return batchMious,batchMdices


# 计算混淆矩阵
def generate_matrix(num_class, gt_image, pre_image):
    # 正确的gt_mask
    mask = (gt_image >= 0) & (gt_image < num_class)  # ground truth中所有正确(值在[0, classe_num])的像素label的mask

    label = num_class * gt_image[mask].astype('int') + pre_image[mask]
    # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    count = np.bincount(label, minlength=num_class ** 2)
    confusion_matrix = count.reshape(num_class, num_class)  # (n, n)
    return confusion_matrix

def miou(hist):

    iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    miou = np.nanmean(iou)
    dice = 2*np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0))
    mdice = np.nanmean(dice)

    a,b,c,d=hist[0][0]/(hist[0][0]+hist[0][1]),hist[0][1]/(hist[0][0]+hist[0][1]),hist[1][0]/(hist[1][0]+hist[1][1]),\
            hist[1][1]/(hist[1][0]+hist[1][1])

    roc1 = np.nanmean(b)

    roc2 = np.nanmean(d)

    SA = np.nanmean((a+d)/2)

    MAE = np.nanmean((b+c)/(a+b+c+d))

    Fb = (a/(a+b)+(a+d)/(a+b+c+d))/2
    return miou,mdice,roc1,roc2,SA,MAE,Fb




import numpy as np
class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        # 找出标签中需要计算的类别,去掉了背景
        mask = (label_true >= 0) & (label_true < self.num_classes)
        # # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    # 输入：预测值和真实值
    # 语义分割的任务是为每个像素点分配一个label
    def evaluate(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            assert len(lp.flatten()) == len(lt.flatten())
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())
        # miou
        iou = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        miou = np.nanmean(iou)
        #mdice
        dice =2.0* np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0))
        mdice = np.nanmean(dice)

        # -----------------其他指标------------------------------
        # mean acc
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.nanmean(np.diag(self.hist) / self.hist.sum(axis=1))

        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()

        return acc, acc_cls, iou, miou,mdice,fwavacc


if __name__ == '__main__':
    for _data_name in ["Kvasir", "ETIS-LaribPolypDB", "CVC-300", "CVC-ClinicDB", "CVC-ColonDB"]:
        data_set_0="M_J_RA_bd_loss"
        data_set_1 = "M_J_RA"
        data_set_2 = "M"
        data_set_3 = "J_RA"
        label_path = r"G:\DeepLearning\deep-learning-for-image-processing-master\pytorch_segmentation\data\息肉1\test\{}\1st_manual".format(_data_name)
        ######修改位置1##########
        predict_path = r"G:\DeepLearning\deep-learning-for-image-processing-master——1\pytorch_segmentation\PRANET\results\PraNet\NEW\Res2Net\多尺度判断传播_凯哥数据集\{}\{}".format(data_set_2,_data_name)

        print(_data_name)

        files = [f for f in os.listdir(label_path) if f.endswith('.jpg') or f.endswith('.png')]
        num_files=len(files)
        mean_iou=0
        mean_iou_1 = 0
        mean_iou_2 = 0
        mean_dice=0
        mean_dice_1=0
        mean_dice_2=0
        mro1 = 0.0
        mro2 = 0.0
        mSA = 0.0
        mFb = 0.0
        mMAE = 0.0
        for i,filename in enumerate(files):
            # print(filename)
            imgLabel = Image.open(label_path +"\\"+ filename)
            imgLabel=imgLabel.convert("RGB")

            # imgPredict = Image.open(predict_path +filename.replace('.png','_res.png'))
            imgPredict = Image.open(predict_path +"\\"+ filename)
            imgPredict=imgPredict.convert("L")
            imgPredict = np.array(imgPredict)  # 可直接换成预测图片
            imgLabel = np.array(imgLabel, dtype='uint8')  # 可直接换成标注图片

            # imgLabel = cv2.imread(label_path +filename)
            # 将数据转为单通道的图片
            imgLabel = cv2.cvtColor(imgLabel, cv2.COLOR_BGR2GRAY)
            imgPredict[imgPredict > 0] = 1
            imgLabel[imgLabel > 0] = 1
        # 计算iou和dice
            #******************
            Miou,Mdice=Iou(imgPredict, imgLabel, classNum=2)
            mean_iou=float(mean_iou)+float(Miou[0])
            mean_dice=float(mean_dice)+float(Mdice[0])

        # 计算iou 和 dice
            EVALUE=IOUMetric(num_classes=2)
            a,b,c,d,e,f=EVALUE.evaluate(imgPredict,imgLabel)
            mean_iou_2 = float(d)+mean_iou_2
            mean_dice_2=float(e)+mean_dice_2
        # 计算iou和dice

            hist = generate_matrix(2, imgLabel, imgPredict)
            miou_res,mdice_res,roc1,roc2,SA,MAE,Fb = miou(hist)
            print("miou_res:",miou_res,"mdice_res",mdice_res)
            mro1 = float(mro1) + float(roc1)

            mro2 = float(mro2) + float(roc2)

            mSA = float(mSA) + float(SA)

            mMAE = float(mMAE) + float(MAE)

            mFb = float(mFb) + float(Fb)

            mean_iou_1 = float(mean_iou_1) + float(miou_res)
            mean_dice_1 = float(mean_dice_1) + float(mdice_res)

            metric = SegmentationMetric(2)  # 2表示有1个分类，有几个分类就填几
            #print(imgPredict.shape, imgLabel.shape)
            metric.addBatch(imgPredict, imgLabel)

            PA = metric.pixelAccuracy()
            CPA = metric.classPixelAccuracy()
            MPA = metric.meanPixelAccuracy()
            IoU = metric.IntersectionOverUnion()
            FWIoU = metric.Frequency_Weighted_Intersection_over_Union()

        print("\n数据集：",_data_name)
        print('像素准确率PA:            %.2f%%' % (PA * 100))
        print('类别平均像素准确率MPA:     %.2f%%' % (MPA * 100))
        print('平均交并比MIoU:           %.2f%%' % (mean_iou/num_files * 100))
        print('平均交并比MIoU:           %.2f%%' % (mean_iou_1/num_files * 100))
        print('平均交并比MIoU:           %.2f%%' % (mean_iou_2/num_files * 100))
        print("The point of ROC:[%.3f,%.3f] "%(mro2/num_files,mro1/num_files))
        print("The data of SA:[%.5f] " % (mSA/num_files))
        print("The data of MAE:[%.5f] " % (mMAE / num_files))
        print("The data of mFb:[%.5f] " % (mFb / num_files))


        print('平均Dice:               %.2f%%' % (mean_dice/num_files * 100))
        print('平均Dice:               %.2f%%' % (mean_dice_1/num_files * 100))
        print('平均Dice:               %.2f%%' % (mean_dice_2/num_files * 100))

        print('权频交并比FWIoU:          %.2f%%' % (FWIoU * 100))
        print('类别像素准确率CPA:        %.2f%%' % (CPA[1] * 100))
        print('交并比IoU:               %.2f%%\n' % (IoU[1] * 100))