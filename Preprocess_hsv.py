# coding = utf-8
import numpy as np
import cv2 as cv
import matplotlib as mpl
from PIL import Image
from scipy import interpolate
import pylab as pl
import math
import os.path
from skimage import data
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
from skimage.color.adapt_rgb import adapt_rgb, each_channel, hsv_value
from skimage import filters

def sobel_each(image):
    return filters.sobel(image)

def sobel_hsv(image):
    return filters.sobel(image)

#读取图像所在的文件夹
rootdir=r'd:/paddle/TEST_SIMPLE/LYMPHOCYTE/'
for parent,dirnames,filenames in os.walk(rootdir):
    for filename in filenames:
     imgpath=os.path.join(parent,filename)
    
     image_stripes = cv.imread(imgpath)
     
     for i in range(1,10) :
        sobel_each(image_stripes)
        sobel_hsv(image_stripes)
     cv.imwrite("E:/TEST_SIMPLE/L" + "/" + filename, image_stripes)
     
print("Complete")    

