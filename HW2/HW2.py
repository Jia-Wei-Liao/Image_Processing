# coding: utf-8

#引入模組
import numpy as np
import cv2


#(A)讀取灰階影像
I = cv2.imread('Gray.jpg', cv2.IMREAD_GRAYSCALE)
n = np.shape(I)[0]
m = np.shape(I)[1]

#Step1:建構dithering matrix
D2 = np.array([[0, 128, 32, 160], [192, 64, 224, 96], [48, 176, 16, 144], [240, 122, 208, 80]])

#用D2貼滿D
D = np.zeros((n,m), np.int)
for i in range(n):
    for j in range(m):
        D[i, j] = D2[(i%4), (j%4)]
I1 = np.zeros((n,m), np.int)

#Step2:
for i in range(n):
    for j in range(m):
        if I[i, j] > D[i, j]:
            I1[i, j] = 255
        else:
            I1[i ,j] = 0

I1 = np.uint8(I1)

#Step3:
# 顯示原圖I
cv2.imshow('My Image1', I)

# 按下任意鍵則關閉所有視窗
cv2.waitKey(0)
cv2.destroyAllWindows()

# 顯示I1
cv2.imshow('My Image2', I1)

# 按下任意鍵則關閉所有視窗
cv2.waitKey(0)
cv2.destroyAllWindows()
Q = np.zeros((n,m), np.int)


#(B)Extend to n = 4 gray values

#1.
N = np.int(255/3)

#2.
Q = np.zeros((n,m), np.int)

for i in range(n):
    for j in range(m):
        Q[i, j] = I[i, j]/N
        
#3.建構dithering matrix
D1 = np.array([[0, 56], [84, 28]])

#extend to D
D = np.zeros((n,m), np.int)
for i in range(n):
    for j in range(m):
        D[i, j] = D1[(i%2), (j%2)]

#4.
I2 = np.zeros((n,m), np.int)

for i in range(n):
    for j in range(m):
        if I[i, j] - N*Q[i, j] > D[i, j]:
            I2[i, j] = Q[i, j] + 1
        else:
            I2[i, j] = Q[i, j]

#5.
for i in range(n):
    for j in range(m):
        I2[i,j] *= N           
I2 = np.uint8(I2)

# 顯示I2
cv2.imshow('My Image2', I2)

# 按下任意鍵則關閉所有視窗
cv2.waitKey(0)
cv2.destroyAllWindows()
Q = np.zeros((n,m), np.int)


#儲存影像
cv2.imwrite('I.jpg', I)
cv2.imwrite('I1.jpg', I1)
cv2.imwrite('I2.jpg', I2)



