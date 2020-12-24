# coding = utf-8
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import matplotlib as mpl
from PIL import Image
from scipy import interpolate
import pylab as pl
import math
import os.path
   
def ft_image(norm_image):
      f = np.fft.fft2(norm_image)
      fshift = np.fft.fftshift(f)
      frequency_tx = 20*np.log(np.abs(fshift))
      return frequency_tx

def make_transform_matrix(image_arr, d0, ftype):
    '''
    构建理想高/低通滤波器
    INPUT  -> 图像数组, 通带半径, 类型
    '''
    transfor_matrix = np.zeros(image_arr.shape, dtype=np.float32)  # 构建滤波器
    w, h = transfor_matrix.shape
    for i in range(w):
        for j in range(h):
            distance = np.sqrt((i - w/2)**2 + (j - h/2)**2)
            if distance < d0:
                transfor_matrix[i, j] = 1
            else:
                transfor_matrix[i, j] = 0
    if ftype == 'low':
        return transfor_matrix
    elif ftype == 'high':
        return 1 - transfor_matrix

namecounter=0
#读取图像所在的文件夹
rootdir=r'D://paddle//SAO'
for parent,dirnames,filenames in os.walk(rootdir):
    for filename in filenames:
     imgpath=os.path.join(parent,filename)
     # 图像读取并灰度化
     image_stripes = cv.imread(imgpath)
     image_stripes = cv.cvtColor(image_stripes, cv.COLOR_BGR2RGB)
     gray_stripes = cv.cvtColor(image_stripes, cv.COLOR_RGB2GRAY)
     image_solid = cv.imread(imgpath)
     image_solid = cv.cvtColor(image_solid, cv.COLOR_BGR2RGB)
     gray_solid = cv.cvtColor(image_solid, cv.COLOR_RGB2GRAY)
     # 为了便于后续处理将颜色空间从 [0,255] 归一化到 [0,1]
     norm_stripes = gray_stripes/255.0
     norm_solid = gray_solid/255.0
     # 图像灰度化
     img_arr = np.array(Image.open(imgpath).convert('L'))
     # 将图像从空间域转换到频率域
     f = np.fft.fft2(img_arr)
     fshift = np.fft.fftshift(f)
     # 生成低通滤波器
     F_filter1 = make_transform_matrix(img_arr, 30, 'low')# 滤波
     result = fshift*F_filter1# 将图像从频率域转换到空间域
     img_d1 = np.abs(np.fft.ifft2(np.fft.ifftshift(result)))
     # 可视化
     #cv.imshow('1',img_arr)
     #cv.waitKey()
     # 要存放生成图片的文件夹
     cv.imwrite("E://after_LUBO" + "/" + filename, img_arr)
     
print("complete")
    

