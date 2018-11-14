#-*-coding:utf-8-*-
'''
Created on Nov 13,2018

@author: pengzhiliang
'''
import numpy as np
import cv2
# 训练时类别
# 0.    Background
# 1.    Cortical gray matter (皮质灰质)
# 2.    Basal ganglia (基底神经节 )
# 3.    White matter (白质)
# 4.    White matter lesions （白质组织）
# 5.    Cerebrospinal fluid in the extracerebral space （脑脊液）
# 6.    Ventricles （脑室）
# 7.    Cerebellum （小脑）
# 8.    Brainstem （脑干）
# 测试时类别，类别合并
# 0.    Background
# 1.    Cerebrospinal fluid (including ventricles)
# 2.    Gray matter (cortical gray matter and basal ganglia)
# 3.    White matter (including white matter lesions)
# label_test:[0,2,2,3,3,1,1,0,0]

# Back 0 : Background 
# GM 2 :   Cortical GM(red), Basal ganglia(green)
# WM 3:   WM(yellow), WM lesions(blue)
# CSF 1 :  CSF(pink), Ventricles(light blue)
# Back: Cerebellum(white), Brainstem(dark red)

color = np.asarray([[0,0,0],[0,0,255],[0,255,0],[0,255,255],[255,0,0],\
        [255,0,255],[255,255,0],[255,255,255],[0,0,128],[0,128,0],[128,0,0]]).astype(np.uint8)
color_test = np.asarray([[0,0,0],[0,0,255],[0,255,0],[255,0,0]]).astype(np.uint8)
# Back , CSF , GM , WM
label_test=[0,2,2,3,3,1,1,0,0]

def merge_classes(label):
    """
    功能：将九类按一定的规则合并成4类，具体间上方注释
    输入： 有9类的二维np.array
    输出： 只有4类的二维np.array
    """
    label = label.astype(np.int)
    label[label == 1] = 2
    label[label == 4] = 3
    label[label == 5] = 1
    label[label == 6] = 1
    label[label == 7] = 0
    label[label == 8] = 0
    return label
def encode(label,color):
    """
    将输入的灰度图转换成RGB图
    输入：
        label： 灰度图（二维 np.array）
        color： 每类对应的颜色 
    """
    H,W = label.shape
    img = np.zeros((H,W,3))
    for i in range(H):
        for j in range(W):
            img[i,j] = color[label[i,j]]
    return img

if __name__ == '__main__':
    a = np.array([[0,1,2,3],[4,5,6,7]],dtype=np.int)
    print(a,'\n',merge_classes(a))
    img = cv2.imread('/home/cv_xfwang/data/MRBrainS/TrainingData/5/slices/5_001.png')
    cv2.imshow('image',img)
    mask = cv2.imread('/home/cv_xfwang/data/MRBrainS/TrainingData/5/slices/5_001_mask.png',0)
    cv2.imshow('Encode mask image',encode(mask,color))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

