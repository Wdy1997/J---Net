# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import roc_curve, auc  ###计算roc和auc

OurNet_M_J_RA_bd_tpr=[0,0.960131,1]
OurNet_M_J_RA_bd_fpr=[0,0.00888,1]

OurNet_M_J_RA_tpr=[0,0.974167,1]
OurNet_M_J_RA_fpr=[0,0.021124,1]

OurNet_M_J_tpr=[0,0.927733,1]
OurNet_M_J_fpr=[0,0.016675,1]

UNet_tpr=[0,0.223,1]
UNet_fpr=[0,0.08133,1]

UNet__tpr=[0,0.962,1]
UNet__fpr=[0,0.064341,1]

RNet__tpr=[0,0.953,1]
RNet__fpr=[0,0.01345,1]


OurNet_M_J_RA_bd_tpr_auc = auc(OurNet_M_J_RA_bd_fpr, OurNet_M_J_RA_bd_tpr)  ###计算auc的值
OurNet_M_J_RA_tpr_auc = auc(OurNet_M_J_RA_fpr, OurNet_M_J_RA_tpr)  ###计算auc的值
OurNet_M_J_tpr_auc = auc(OurNet_M_J_fpr, OurNet_M_J_tpr)  ###计算auc的值
Unet_roc_auc = auc(UNet_fpr, UNet_tpr)  ###计算auc的值
Unet__roc_auc = auc(UNet__fpr, UNet__tpr)  ###计算auc的值
Rnet__roc_auc = auc(RNet__fpr, RNet__tpr)  ###计算auc的值


lw = 2
alp=0.5
plt.figure(figsize=(5, 5))
plt.plot(OurNet_M_J_RA_bd_fpr, OurNet_M_J_RA_bd_tpr, color='darkorange', alpha=alp,lw=lw, label='J-Net(M+J_RA+bd_loss) curve (area = %0.3f)' % OurNet_M_J_RA_bd_tpr_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot(OurNet_M_J_RA_fpr, OurNet_M_J_RA_tpr, color='teal', alpha=alp,lw=lw, label='J-Net(M+J_RA) curve (area = %0.3f)' % OurNet_M_J_RA_tpr_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot(OurNet_M_J_fpr, OurNet_M_J_tpr, color='m', alpha=alp,lw=lw, label='J-Net(M) curve (area = %0.3f)' % OurNet_M_J_tpr_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot(UNet_fpr, UNet_tpr, color='blue', alpha=alp,lw=lw, label='UNet curve (area = %0.3f)' % Unet_roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot(UNet__fpr, UNet__tpr, color='black', alpha=alp,lw=lw, label='UNet++ curve (area = %0.3f)' % Unet__roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot(RNet__fpr, RNet__tpr, color='red', alpha=alp,lw=lw, label='ResNet++ curve (area = %0.3f)' % Rnet__roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve of proposed models on the CVC_ClinicDB')
plt.legend(loc="lower right")
plt.show()