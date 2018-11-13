#-*-coding:utf-8-*-
'''
Created on Nov 13,2018

@author: pengzhiliang
'''

# 训练时类别
# 0.	Background
# 1.	Cortical gray matter (皮质灰质)
# 2.	Basal ganglia (基底神经节 )
# 3.	White matter (白质)
# 4.	White matter lesions （白质组织）
# 5.	Cerebrospinal fluid in the extracerebral space （脑脊液）
# 6.	Ventricles （脑室）
# 7.	Cerebellum （小脑）
# 8.	Brainstem （脑干）
# 测试时类别，类别合并
# 0.	Background
# 1.	Cerebrospinal fluid (including ventricles)
# 2.	Gray matter (cortical gray matter and basal ganglia)
# 3.	White matter (including white matter lesions)
# label_test:[0,2,2,3,3,1,1,0,0]

# Back 0 : Background 
# GM 2 :   Cortical GM(red), Basal ganglia(green)
# WM 3:   WM(yellow), WM lesions(blue)
# CSF 1 :  CSF(pink), Ventricles(light blue)
# Back: Cerebellum(white), Brainstem(dark red)

color=np.asarray([[0,0,0],[0,0,255],[0,255,0],[0,255,255],[255,0,0],\
        [255,0,255],[255,255,0],[255,255,255],[0,0,128],[0,128,0],[128,0,0]]).astype(np.uint8)
# Back , CSF , GM , WM
label_test=[0,2,2,3,3,1,1,0,0]