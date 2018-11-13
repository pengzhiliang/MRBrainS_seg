#-*-coding:utf-8-*-
'''
Created on Nov 13,2018

@author: pengzhiliang
'''
import os
import os.path as osp
import numpy as np

import torch
from torch.utils import data
import cv2

class MRBrainSDataset(data.Dataset):
    def __init__(self, root, split='train', is_transform=True, augmentations=None, img_norm=False):
        self.root = root
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        files = tuple(open(osp.join(root, split+'.txt'), 'r'))
        self.files = [file_.rstrip('\n') for file_ in files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        img = cv2.imread(osp.join(self.root, self.files[index]+'.png'))
        mask = cv2.imread(osp.join(self.root, self.files[index]+'_mask.png'), 0)

        if self.augmentations is not None:
            img, mask = self.augmentations(img, mask)
        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mask = mask.astype(np.uint8)

        if self.img_norm:
            img = img / 255.0

        if self.is_transform:
            # to transpose, to Tensor
            img, mask = self._transform(img, mask)
        else:
            img = data_np.astype(np.float32)

        return img, mask

    def _transform(self, img, mask):
        img = img.astype(np.float64)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()
        return img, mask


if __name__ == '__main__':
    from augmentations import *
    # from utils.utils import *
    # data_aug=None
    data_aug = Compose([
               RandomHorizontallyFlip(0.5),
               RandomRotate(10),
               # Scale(256),
               ])

    dloader = torch.utils.data.DataLoader(MRBrainSDataset(osp.join('/home/cv_xfwang/data/', 'MRBrainS'), split='train', is_transform=True, img_norm=True, augmentations=data_aug), batch_size=1)
    for idx, (img, mask) in enumerate(dloader):
        if idx < 10:
            img = img.cpu().data[0].numpy().transpose(1, 2, 0)
            mask = mask.cpu().data[0].numpy()
            #print(mask,np.sum(mask))
            img = img * 255.0
            import cv2
            #cv2.imwrite('sample/%d.png'%(idx+1), img.astype(np.uint8))
            cv2.imshow('sample image',img.astype(np.uint8))
            print(np.unique(mask))
            #cv2.imwrite('sample/%d_mask.png'%(idx+1), mask.astype(np.uint8))
            cv2.imshow('sample mask', mask.astype(np.uint8))
            cv2.waitKey(0)
        else:
            exit(-1)
