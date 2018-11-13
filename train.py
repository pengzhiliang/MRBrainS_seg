#-*-coding:utf-8-*-
'''
Created on Oct 31,2018

@author: pengzhiliang
'''

import time 
import numpy as np
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset,DataLoader
from torch.optim import lr_scheduler,Adam,SGD
from torchvision import datasets, models, transforms
from torchsummary import summary
from model import fcn,fcn_dilated
from utils import Score,averageMeter,cross_entropy2d
from dataloader.MRBrain_loder import MRBrainSDataset
from dataloader.augmentation import * 

# 参数设置
defualt_path = osp.join('/home/cv_xfwang/data/', 'MRBrainS')
learning_rate = 1e-8
batch_size = 32
num_workers = 4
resume_path = '/home/cv_xfwang/MRBrainS_seg/checkpoint/best_model.pkl'
resume_flag = True
start_epoch = 0
end_epoch = 1000
test_interval = 10
print_interval = 1
momentum=0.99
weight_decay = 0.005
best_iou = -100

# GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup Dataloader
data_aug = Compose([
           RandomHorizontallyFlip(0.5),
           RandomRotate(10),
           Scale(256),
           ])
train_loader = DataLoader(MRBrainSDataset(defualt_path, split='train', is_transform=True, \
                img_norm=True, augmentations=data_aug), \
                batch_size=batch_size,num_workers=num_workers,pin_memory=True,shuffle=True)
val_loader = DataLoader(MRBrainSDataset(defualt_path, split='val', is_transform=True, \
                img_norm=True, augmentations=None), \
                batch_size=batch_size,num_workers=num_workers,pin_memory=True,shuffle=False)

# Setup Model and summary
model = model(n_classes=9).to(device)
vgg16 = models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("/home/cv_xfwang/pretrained/vgg16-397923af.pth"))
model.init_vgg16_params(vgg16)
# model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
summary(model,(3,256,256)) # summary 网络参数

# 需要学习的参数
# base_learning_list = list(filter(lambda p: p.requires_grad, model.base_net.parameters()))
# learning_list = model.parameters()

# 优化器以及学习率设置
# optimizer = SGD([
#                     {'params': model.base_net.parameters(),'lr': learning_rate / 10},
#                     {'params': model.model_class.parameters(), 'lr': learning_rate * 10},
#                     {'params': model.model_reg.parameters(), 'lr': learning_rate * 10}
#                 ], lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=alearning_rate,  
    momentum=momentum, 
    weight_decay=weight_decay 
)

scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.4 * end_epoch), int(0.7 * end_epoch),int(0.8 * end_epoch),int(0.9 * end_epoch)], gamma=0.1)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',patience=10, verbose=True)
criterion = cross_entropy2d()

# resume 
if (os.path.isfile(resume_path) and resume_flag):
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    best_iou = checkpoint['best_iou']
    # scheduler.load_state_dict(checkpoint["scheduler_state"])
    # start_epoch = checkpoint["epoch"]
    print("=====>",
        "Loaded checkpoint '{}' (iter {})".format(
            resume_path, checkpoint["epoch"]
        )
    )
else:
    print("=====>","No checkpoint found at '{}'".format(resume_path))

# Training
def train(epoch):
    print("Epoch: ",epoch)
    model.train()
    total_loss = 0
    for index, (img, mask) in enumerate(train_loader):
        img = img.to(device)
        optimizer.zero_grad()
        output = model(img) #[-1, 9, 256, 256]
        _, pred = torch.max(output, dim=1)
        loss = criterion(output,mask)
        total_loss += loss
        loss.backward()
        optimizer.step()

        print("loss: %.4f"%(total_loss/(img.szie(0)*(index+1))) )

# return mean IoU, mean dice
def test(epoch):
    print(">>>Test: ")
    global best_iou
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for i, (img, mask) in val_loader:
            img = img.to(device)
            output = model(img)
            output = F.interpolate(output, size=(h, w),  mode='bilinear', align_corners=True)
            probs = F.softmax(output, dim=1)
            _, pred = torch.max(probs, dim=1)
            pred = pred.cpu().data[0].numpy()

            label = mask.cpu().data[0].numpy()
            pred = np.asarray(pred, dtype=np.int)
            label = np.asarray(label, dtype=np.int)
            gts.append(label)
            preds.append(preds)

    whole_brain_preds = np.dstack(preds)
    whole_brain_gts = np.dstack(gts)
    running_metrics = Score(9)
    running_metrics.update(whole_brain_gts, whole_brain_preds)
    scores, class_iou = running_metrics.get_scores()
    mIoU = np.nanmean(class_iou[1::])
    mean_dice = (mIoU * 2) / (mIoU + 1)
    
    print("mean Iou",mIoU, "mean dice",mean_dice)
    if mIoU > best_iou:
        best_iou = mIoU
        state = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_iou": best_iou,
        }
        save_path = osp.join(osp.split(resume_path)[0],"best_model.pkl")
        print("saving......")
        torch.save(state, save_path)
    return mIoU, mean_dice


for epoch in range(start_epoch, end_epoch):
    train(epoch)
    test(epoch)
    scheduler.step()
    # print(train_loss[-1],train_acc[-1],test_loss[-1],test_acc[-1]