# coding: utf-8
import numpy as np
import cv2

# 顯示圖片
def imgshow(img):
    cv2.imshow('My Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Convolution
def conv(Matrix, Filter):
    Matrix = np.pad(Matrix,((1,1),(1,1)),'constant',constant_values = (0,0))
    n, m = np.shape(Matrix)
    Matrix2 = np.zeros((n, m)).astype('double')

    for i in range(1, n-1):
        for j in range(1 ,m-1):
            Matrix2[i,j] =  (Matrix[i-1:i+2, j-1:j+2]*Filter).sum()

    Matrix2 = Matrix2.astype('uint8')
    return Matrix2[1:n-1, 1:m-1]

# Step 1: Choose a grayscale image I
I = cv2.imread('Gray.jpg', cv2.IMREAD_GRAYSCALE)
n, m = np.shape(I)

# Step 2: Zero interleave
I1 = np.zeros((n*2,m*2), np.double)

for i in range(n*2):
    for j in range(m*2):
        if i%2==0 and j%2==0:
            I1[i, j] = I[i//2, j//2]

# Step 3: Fill values by convolving I1 with
# (i) NN interpolation
NN = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])

# (ii) Bilinear interpolation
Bilinear = (1/4)*np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])

img_NN = conv(I1, NN)
img_Bilinear = conv(I1, Bilinear)

img1 = np.hstack((img_NN, img_Bilinear)) 
imgshow(img1)

cv2.imwrite('img_NN.jpg', img_NN)
cv2.imwrite('img_Bilinear.jpg', img_Bilinear)
