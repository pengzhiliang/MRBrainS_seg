import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import Variable

def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h > ht and w > wt: # upsample labels
        target = target.unsequeeze(1)
        target = F.upsample(target, size=(h, w), mode='nearest')
        target = target.sequeeze(1)
    elif h < ht and w < wt: # upsample images
        input = F.upsample(input, size=(ht, wt), mode='bilinear')
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")

    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.contiguous().view(-1, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, ignore_index=250,
                      weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss

def dice_loss(input, target):
    target=target.cpu().unsqueeze(1)
    # target_bin=Variable(torch.zeros(1,9,target.shape[2],target.shape[3])).scatter_(1,target,1).cuda()
    target_bin=torch.zeros(1,9,target.shape[2],target.shape[3]).scatter_(1,target,1).cuda()
    target=target.squeeze(1).cuda()
    target=target.squeeze(1).cuda()
    smooth = 1.

    iflat = input.view(-1)
    tflat = target_bin.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

def weighted_loss(input,target_bin,weight,size_average=True):
    n,c,h,w=input.size()
    # NHWC
    input=F.softmax(input,dim=1).transpose(1,2).transpose(2,3).contiguous().view(-1,c)
    input=input[target_bin.view(n*h*w,c)>=0]
    input=input.view(-1,c)

    weight=weight.transpose(1,2).transpose(2,3).contiguous()
    weight=weight.view(n*h*w,1).repeat(1,c)
    '''
    mask=target>=0
    target=target[mask]
    target_bin=np.zeros((n*h*w,c),np.float)
    for i,term in enumerate(target):
        target_bin[i,int(term)]=1
    target_bin=torch.from_numpy(target_bin).float()
    target_bin=Variable(target_bin.cuda())
    '''
    loss=F.binary_cross_entropy(input,target_bin,weight=weight,size_average=False)
    if size_average:
        loss/=(target_bin>=0).data.sum()/c
    return loss

def bootstrapped_cross_entropy2d(input, target, K, weight=None, size_average=True):

    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input, target, K, weight=None, size_average=True):
        n, c, h, w = input.size()
        log_p = F.log_softmax(input, dim=1)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
        log_p = log_p.view(-1, c)

        mask = target >= 0
        target = target[mask]
        loss = F.nll_loss(log_p, target, weight=weight, ignore_index=250,
                          reduce=False, size_average=False)
        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(input=torch.unsqueeze(input[i], 0),
                                           target=torch.unsqueeze(target[i], 0),
                                           K=K,
                                           weight=weight,
                                           size_average=size_average)
    return loss / float(batch_size)