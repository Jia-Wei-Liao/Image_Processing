# coding: utf-8
import numpy as np
import cv2
import random

# 讀取灰階影像
I = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

#判斷影像的矩陣大小
n = np.shape(I)[0]
m = np.shape(I)[1]

# 顯示圖片
def imgshow(img):
    cv2.imshow('My Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 周圍補0，確保每個元素都能被filter掃到
I1 = np.pad(I,((1,1),(1,1)),'constant',constant_values = (0,0))
N = np.shape(I1)[0]
M = np.shape(I1)[1]

# Average filter 和 Median filter
I2 = np.zeros((N, M), int).astype('double')
I3 = np.zeros((N, M), int).astype('uint8')

for i in range(1, N-1):
    for j in range(1 ,M-1):
        I2[i,j] =  I1[i-1:i+2, j-1:j+2].mean()
        I3[i,j] =  sorted(I1[i-1:i+2, j-1:j+2].reshape(9))[4]

I2 = I2.astype('uint8')
I3 = I3.astype('uint8')

# Unsharp masking
k = 0.4
s = 1/(1-k)
I4 = ((I1 - k*I2)*s).astype('uint8')
I5 = ((I1 - k*I3)*s).astype('uint8')
img1 = np.hstack((I1, I2, I3)) 
img2 = np.hstack((I1, I4, I5)) 

imgshow(img1)
imgshow(img2)
cv2.imwrite('input_image.jpg', I)
cv2.imwrite('average_filter.jpg', I2)
cv2.imwrite('median_filter.jpg', I3)
cv2.imwrite('average_filter_us.jpg', I4)
cv2.imwrite('median_filter_us.jpg', I5)
