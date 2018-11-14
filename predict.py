#-*-coding:utf-8-*-
'''
Created on Nov 14,2018

@author: pengzhiliang
'''

import time 
import numpy as np
import cv2
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from model.fcn import fcn,fcn_dilated
from model.unet import UNet
from dataloader.augmentation import * 
from dataloader.coder import *
from utils.crf import dense_crf

# 图片预处理
def transform(img, mask):
    img = img/255.0
    img = img.astype(np.float64)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    mask = torch.from_numpy(mask).long()
    return img, mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 从 fcn,UNet中任选一个模型
model = UNet().to(device)
resume_path = './checkpoint/best_'+'unet'+'_model.pkl' #fcn,unet
root_path = './image'
img_path = root_path+'.png'
mask_path = root_path+'_mask.png'

image = cv2.imread(img_path)
mask = cv2.imread(mask_path, 0)
img,mask = Compose([Scale(224)])(image.copy(),mask)
# image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
mask = mask.astype(np.uint8)
img, mask = transform(img, mask)
img, mask = torch.unsqueeze(img,0), torch.unsqueeze(mask,0)
# resume 
if osp.isfile(resume_path) :
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint["model_state"])
    best_iou = checkpoint['best_iou']
    print("=====>",
        "Loaded checkpoint '{}' (iter {})".format(
            resume_path, checkpoint["epoch"]
        ) )
    print("=====> best mIoU: %.4f best mean dice: %.4f"%(best_iou,(best_iou*2)/(best_iou+1)))
   
else:
    raise ValueError("can't find model")

crf = True

with torch.no_grad():
    img,mask= img.to(device),mask.to(device)
    output = model(img) #[1, 9, 256, 256]
    probs = F.softmax(output, dim=1)
    if crf:
        pred_crf = probs.cpu().data[0].numpy()
        # crf
        img = img.cpu().data[0].numpy()
        pred_crf = dense_crf(img*255, pred_crf)
        pred_crf = np.asarray(pred_crf, dtype=np.int)
        # 合并特征
        pred_crf = merge_classes(pred_crf)
    _, pred = torch.max(probs, dim=1)
    pred = pred.cpu().data[0].numpy()          
    label = mask.cpu().data[0].numpy()
    pred = np.asarray(pred, dtype=np.int)
    label = np.asarray(label, dtype=np.int)
    pred = merge_classes(pred)
    label = merge_classes(label)
cv2.namedWindow("image",0)
cv2.imshow("image",image)
cv2.namedWindow("mask",0)
cv2.imshow("mask",encode(label,color_test))
cv2.namedWindow("pred",0)
cv2.imshow("pred",encode(pred,color_test))
cv2.namedWindow("pred_crf",0)
cv2.imshow("pred_crf",encode(pred_crf,color_test))
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("image.png",image)
cv2.imwrite("mask.png",encode(label,color_test))
cv2.imwrite("pred_mask.png",encode(pred,color_test))
cv2.imwrite("pred_crf_mask.png",encode(pred_crf,color_test))
