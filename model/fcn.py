#-*-coding:utf-8-*-
'''
Created on Nov 13,2018

@author: pengzhiliang
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class fcn(nn.Module):
    def __init__(self,n_classes=9):
        super(fcn, self).__init__()
        self.n_classes = n_classes

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),nn.ReLU(inplace=True),)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),nn.ReLU(inplace=True),)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),nn.ReLU(inplace=True),)
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),nn.ReLU(inplace=True),)
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),nn.ReLU(inplace=True),)

        self.pool=nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv1_16=nn.Conv2d(64, 16, 3, padding=1)
        self.conv2_16=nn.Conv2d(128, 16, 3, padding=1)
        self.up_conv2_16 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.conv3_16=nn.Conv2d(256, 16, 3, padding=1)
        self.up_conv3_16 = nn.ConvTranspose2d(16, 16, kernel_size=4, stride=4)
        self.conv4_16=nn.Conv2d(512, 16, 3, padding=1)
        self.up_conv4_16 = nn.ConvTranspose2d(16, 16, kernel_size=8, stride=8)
        self.conv5_16=nn.Conv2d(512, 16, 3, padding=1)
        self.up_conv5_16 = nn.ConvTranspose2d(16, 16, kernel_size=16, stride=16)

        self.score=nn.Sequential(
            nn.Conv2d(4*16,self.n_classes,1),
            nn.Dropout(0.5),
            )

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(self.pool(conv1))
        conv3 = self.conv_block3(self.pool(conv2))
        conv4 = self.conv_block4(self.pool(conv3))
        conv5 = self.conv_block5(self.pool(conv4))
        
        conv1_16=self.conv1_16(conv1)
        up_conv2_16=self.up_conv2_16(self.conv2_16(conv2))
        up_conv3_16=self.up_conv3_16(self.conv3_16(conv3))
        up_conv4_16=self.up_conv4_16(self.conv4_16(conv4))
        up_conv5_16=self.up_conv5_16(self.conv5_16(conv5))

        concat_1_to_4=torch.cat([conv1_16,up_conv2_16,up_conv3_16,up_conv4_16], 1)
        score=self.score(concat_1_to_4)
        return score # [-1, 9, 256, 256]

    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [self.conv_block1,
                  self.conv_block2,
                  self.conv_block3,
                  self.conv_block4,
                  self.conv_block5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data


class fcn_dilated(nn.Module):
    def __init__(self,n_classes=9):
        super(fcn_dilated, self).__init__()
        self.n_classes = n_classes

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, dilation=1, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, dilation=2, padding=2),nn.ReLU(inplace=True),)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, dilation=1, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, dilation=2, padding=2),nn.ReLU(inplace=True),)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, dilation=1, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, dilation=2, padding=2),nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, dilation=3, padding=3),nn.ReLU(inplace=True),)
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, dilation=1, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2),nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=3, padding=3),nn.ReLU(inplace=True),)
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, dilation=1, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2),nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=3, padding=3),nn.ReLU(inplace=True),)

        self.conv1_16=nn.Conv2d(64, 16, 3, padding=1)
        self.conv2_16=nn.Conv2d(128, 16, 3, padding=1)
        self.conv3_16=nn.Conv2d(256, 16, 3, padding=1)
        self.conv4_16=nn.Conv2d(512, 16, 3, padding=1)
        self.conv5_16=nn.Conv2d(512, 16, 3, padding=1)

        self.score=nn.Sequential(
            nn.Conv2d(4*16,self.n_classes,1),
            nn.Dropout(0.5),
            )

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        conv1_16=self.conv1_16(conv1)
        conv2_16=self.conv2_16(conv2)
        conv3_16=self.conv3_16(conv3)
        conv4_16=self.conv4_16(conv4)
        conv5_16=self.conv5_16(conv5)

        concat_1_to_4=torch.cat([conv1_16,conv2_16,conv3_16,conv4_16], 1)
        score=self.score(concat_1_to_4)
        return score

    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [self.conv_block1,
                  self.conv_block2,
                  self.conv_block3,
                  self.conv_block4,
                  self.conv_block5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data

if __name__=='__main__':
    # x=torch.Tensor(4,3,256,256)
    # model=fcn(n_classes=9)
    # y=model(x)
    # print(y.shape)
    model=fcn(n_classes=9)
    summary(model.cuda(),(3,256,256))