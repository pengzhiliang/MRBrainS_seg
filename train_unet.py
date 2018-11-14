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
from utils.metrics import Score,averageMeter
from utils.loss import cross_entropy2d,BCEDiceLoss,bootstrapped_cross_entropy2d
from dataloader.MRBrain_loader import MRBrainSDataset
from dataloader.augmentation import * 
from dataloader.coder import merge_classes


# 参数设置
defualt_path = osp.join('/home/cv_xfwang/data/', 'MRBrainS')
learning_rate = 1e-6
batch_size = 32
num_workers = 4
resume_path = '/home/cv_xfwang/MRBrainS_seg/checkpoint/best_unet_model.pkl'
resume_flag = True
start_epoch = 0
end_epoch = 500
test_interval = 10
print_interval = 1
momentum=0.99
weight_decay = 0.005
best_iou = -100

# GPU or CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup Dataloader
data_aug = Compose([
           RandomHorizontallyFlip(0.5),
           RandomRotate(10),
           Scale(224),
           ])
train_loader = DataLoader(MRBrainSDataset(defualt_path, split='train', is_transform=True, \
                img_norm=True, augmentations=data_aug), \
                batch_size=batch_size,num_workers=num_workers,pin_memory=True,shuffle=True)
val_loader = DataLoader(MRBrainSDataset(defualt_path, split='val', is_transform=True, \
                img_norm=True, augmentations=Compose([Scale(224)])), \
                batch_size=1,num_workers=num_workers,pin_memory=True,shuffle=False)

# Setup Model and summary
model = UNet().to(device)
summary(model,(3,224,224),batch_size) # summary 网络参数
# model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))


# 需要学习的参数
# base_learning_list = list(filter(lambda p: p.requires_grad, model.base_net.parameters()))
# learning_list = model.parameters()

# 优化器以及学习率设置
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=momentum,weight_decay=weight_decay)
# learning rate调节器
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.2 * end_epoch), int(0.6 * end_epoch),int(0.9 * end_epoch)], gamma=0.01)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',patience=10, verbose=True)
criterion = cross_entropy2d
# criterion = BCEDiceLoss()

# running_metrics = Score(n_classes=9)
running_metrics = Score(n_classes=4) # label_test=[0,2,2,3,3,1,1,0,0]
label_test = [0,2,2,3,3,1,1,0,0]
# resume 
if (osp.isfile(resume_path) and resume_flag):
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
    print("load unet weight and bias")
    model_dict = model.state_dict()
    pretrained_dict = torch.load("/home/cv_xfwang/Pytorch-UNet/MODEL.pth")
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

# Training
def train(epoch):
    print("Epoch: ",epoch)
    model.train()
    total_loss = 0
    # for index, (img, mask) in tqdm(enumerate(train_loader)):
    for index, (img, mask) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Epoch {}".format(epoch), ncols=0):
        #img: torch.Size([32, 3, 256, 256]) mask:torch.Size([32, 256, 256])
        img,mask= img.to(device),mask.to(device)
        optimizer.zero_grad()
        output = model(img) #[-1, 9, 256, 256]
        # _, pred = torch.max(output, dim=1)
        loss = criterion(output,mask)#,size_average=False
        total_loss += loss
        loss.backward()
        optimizer.step()

    print("Average loss: %.4f"%(total_loss/(img.size(0)*(index+1))) )

# return mean IoU, mean dice
def test(epoch):
    print(">>>Test: ")
    global best_iou
    model.eval()
    running_metrics.reset()
    with torch.no_grad():
        for i, (img, mask) in tqdm(enumerate(val_loader)):
            img = img.to(device)
            output = model(img) #[-1, 9, 256, 256]
            probs = F.softmax(output, dim=1)
            _, pred = torch.max(probs, dim=1)
            pred = pred.cpu().data[0].numpy()          
            label = mask.cpu().data[0].numpy()
            pred = np.asarray(pred, dtype=np.int)
            label = np.asarray(label, dtype=np.int)
            # print(pred.shape,label.shape)

            running_metrics.update(merge_classes(label),merge_classes(pred))

    score, class_iou = running_metrics.get_scores()
    for k, v in score.items():
        print(k,':',v)
    print(i, class_iou)
    if score["Mean IoU : \t"] > best_iou:
        best_iou = score["Mean IoU : \t"]
        state = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_iou": best_iou,
        }
        save_path = osp.join(osp.split(resume_path)[0],"best_unet_model.pkl")
        print("saving......")
        torch.save(state, save_path)
    # return mIoU, mean_dice


for epoch in range(start_epoch, end_epoch):
    train(epoch)
    test(epoch)
    scheduler.step()
    # print(train_loss[-1],train_acc[-1],test_loss[-1],test_acc[-1]