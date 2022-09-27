# coding: utf-8

#引入模組
import numpy as np
import cv2

# 讀取灰階影像
I = cv2.imread('Gray.jpg', cv2.IMREAD_GRAYSCALE)

#判斷影像的矩陣大小
n = np.shape(I)[0]
m = np.shape(I)[1]
E = np.zeros((n,m), np.double)
I = np.double(I)

#For each pixel I(x,y)
for i in range(n):
    for j in range(m):
        
        #1)Calculate quantization error
        if I[i, j] >= 128:
            E[i, j] = I[i, j]-255
        else:
            E[i, j] = I[i, j]
            
        #2)Floyd-Steinberg    
        if j+1<=m-1:
            I[i, j+1] += (7/16)*E[i, j]
        if i+1<=n-1 and j-1>=0:
            I[i+1, j-1] += (3/16)*E[i, j]
        if i+1<=n-1:
            I[i+1, j] += (5/16)*E[i, j]
        if i+1<=n-1 and j+1<=m-1:
            I[i+1, j+1] += (1/16)*E[i, j]

#3)Quantize new I(x,y) to 0 or 255 using 128 as the threshold
for i in range(n):
    for j in range(m):
        if I[i, j]>=128:
            I[i, j] = 1
        else:
            I[i, j] = 0
I = I*255           
I = np.uint8(I)

cv2.imwrite('Floyd-Steinberg.jpg', I)
