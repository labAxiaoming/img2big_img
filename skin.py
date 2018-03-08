# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 12:52:06 2018

@author: 123
"""

from PIL import Image,ImageDraw,ImageFont ,ImageEnhance
from PIL import ImageFile                #加这两个
ImageFile.LOAD_TRUNCATED_IMAGES = True   #否则导入图片会错误
from skimage import io
import numpy as np
from numpy import *
import cv2
print('wait amount')

bg=io.imread('bgg.jpg')   #background img
bg=Image.fromarray(bg)    #转化成Image类型
bg2=bg.resize((4800,4000), Image.ANTIALIAS)  #resize成4800*4000大小
bg2=np.array(bg2)                           #转化回nparray类型
bg2=uint8(bg2)
#io.imshow(bg2)
#50 100 
#20,12
str1='I:/Spyderdemo/cv2/lolskin/skin'+ '/*.jpg'  #读取文件
col = io.ImageCollection(str1)
leng=len(col)
traindata=np.zeros([240,15000])#数据
trainty=np.zeros([240,1])#标签
for i in range (240):                    #把读取到的图片变成 100*50*3像素大小的
    m=col[i]
    m2=Image.fromarray(m)
    m3=m2.resize((100,50),Image.ANTIALIAS)
    m4=np.array(m3)
    m4=m4.reshape(1,15000)              #把读取到的图片变成 100*50*3像素大小的 然后reshape成1行15000列
    traindata[i,:]=m4                   #traindata第i行  变成上面这一行   1行15000列的数据
    trainty[i]=i                        #第i个标签
#    cv2.imwrite('I:/Spyderdemo/cv2/lolskin/skin2/%d.jpg'%(i),m2)
#    io.imsave('I:/Spyderdemo/cv2/lolskin/skin2/%d.jpg'%(i),m4)

from sklearn.neighbors import KNeighborsClassifier  #机器学习3行代码
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(traindata,trainty)             #训练


testdata=np.zeros([3840,15000])     #把上面背景图片4800*4000*3像素大图变成 100*50*3像素大小的   48*80个方块  
for gao in range (80):
    for kuan in range(48):
        testn=bg2[50*gao:50*gao+50,100*kuan:100*kuan+100]
        testdata[gao*48+kuan]=testn.reshape(1,15000)                  #每个方块（100*50*3）reshape变成 1行15000列
        
testdata=uint8(testdata)
pre_label=knn.predict(testdata)                                       #机器学习预测predict
pre_label=uint8(pre_label)     #找到标签

bg3=bg2.copy()                 #下面这些是根据预测的标签  把背景图每个小方块  替换成相应小图
for gao in range (80):
    for kuan in range(48):
        i=gao*48+kuan
        bg3[50*gao:50*gao+50,100*kuan:100*kuan+100]=traindata[pre_label[i]].reshape(50,100,3)
    
io.imshow(bg3)  

io.imsave('bgg4.jpg',bg3)  #保存


















