#-*-coding:utf-8-*-
'''
Created on Nov14 31,2018

@author: pengzhiliang
'''

import time 
import numpy as np
import os
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from torch.optim import lr_scheduler,Adam,SGD
from torchvision import datasets, models, transforms
from torchsummary import summary
from model.unet import UNet
from model.fcn import fcn
from utils.metrics import Score,averageMeter
from utils.crf import dense_crf
from dataloader.MRBrain_loader import MRBrainSDataset
from dataloader.augmentation import * 
from dataloader.coder import merge_classes


# GPU or CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 参数设置
defualt_path = osp.join('/home/cv_xfwang/data/', 'MRBrainS')
batch_size = 1
num_workers = 4
resume_path = '/home/cv_xfwang/MRBrainS_seg/checkpoint/best_unet_model.pkl'
# data loader
val_loader = DataLoader(MRBrainSDataset(defualt_path, split='val', is_transform=True, \
                img_norm=True, augmentations=Compose([Scale(224)])), \
                batch_size=1,num_workers=num_workers,pin_memory=True,shuffle=False)
# Setup Model and summary
model = UNet().to(device)
# summary(model,(3,224,224),batch_size) # summary 网络参数

# running_metrics = Score(n_classes=9)
running_metrics = Score(n_classes=4) # label_test=[0,2,2,3,3,1,1,0,0]
# resume 
if osp.isfile(resume_path):
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint["model_state"])
    best_iou = checkpoint['best_iou']
    print("=====>",
        "Loaded checkpoint '{}' (iter {})".format(
            resume_path, checkpoint["epoch"]
        ))
    print("=====> best mIoU: %.4f best mean dice: %.4f"%(best_iou,(best_iou*2)/(best_iou+1)))
else:
    raise ValueError("can't find model")


print(">>>Test After Dense CRF: ")
model.eval()
running_metrics.reset()
with torch.no_grad():
    for i, (img, mask) in tqdm(enumerate(val_loader)):
        img = img.to(device)
        output = model(img) #[-1, 9, 256, 256]
        probs = F.softmax(output, dim=1)
        pred = probs.cpu().data[0].numpy()
        label = mask.cpu().data[0].numpy()
        # crf
        img = img.cpu().data[0].numpy()
        pred = dense_crf(img*255, pred)
        # print(pred.shape)
        # _, pred = torch.max(torch.tensor(pred), dim=-1)
        pred = np.asarray(pred, dtype=np.int)
        label = np.asarray(label, dtype=np.int)
        # 合并特征
        pred = merge_classes(pred)
        label = merge_classes(label)
        # print(pred.shape,label.shape)
        running_metrics.update(label,pred)

score, class_iou = running_metrics.get_scores()
for k, v in score.items():
    print(k,':',v)
print(i, class_iou)