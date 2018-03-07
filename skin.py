# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 12:52:06 2018

@author: 123
"""

from PIL import Image,ImageDraw,ImageFont ,ImageEnhance
from PIL import ImageFile                #加这两个
ImageFile.LOAD_TRUNCATED_IMAGES = True   #否则错误
from skimage import io
import numpy as np
from numpy import *
import cv2
print('wait amount')

bg=io.imread('bgg.jpg')   #background img
bg=Image.fromarray(bg)
bg2=bg.resize((4800,4000), Image.ANTIALIAS)
bg2=np.array(bg2)
bg2=uint8(bg2)
#io.imshow(bg2)
#50 100 
#20,12
str1='I:/Spyderdemo/cv2/lolskin/skin'+ '/*.jpg'
col = io.ImageCollection(str1)
leng=len(col)
traindata=np.zeros([240,15000])#数据
trainty=np.zeros([240,1])#标签
for i in range (240):
    m=col[i]
    m2=Image.fromarray(m)
    m3=m2.resize((100,50),Image.ANTIALIAS)
    m4=np.array(m3)
    m4=m4.reshape(1,15000)
    traindata[i,:]=m4
    trainty[i]=i
#    cv2.imwrite('I:/Spyderdemo/cv2/lolskin/skin2/%d.jpg'%(i),m2)
#    io.imsave('I:/Spyderdemo/cv2/lolskin/skin2/%d.jpg'%(i),m4)

from sklearn.neighbors import KNeighborsClassifier  #机器学习3行代码
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(traindata,trainty)


testdata=np.zeros([3840,15000])#数据
for gao in range (80):
    for kuan in range(48):
        testn=bg2[50*gao:50*gao+50,100*kuan:100*kuan+100]
        testdata[gao*48+kuan]=testn.reshape(1,15000)
        
testdata=uint8(testdata)
pre_label=knn.predict(testdata)
pre_label=uint8(pre_label)

bg3=bg2.copy()
for gao in range (80):
    for kuan in range(48):
        i=gao*48+kuan
        bg3[50*gao:50*gao+50,100*kuan:100*kuan+100]=traindata[pre_label[i]].reshape(50,100,3)
    
io.imshow(bg3)

io.imsave('bgg4.jpg',bg3)


















