# coding: utf-8
import numpy as np
import cv2

# 讀取圖檔
img = cv2.imread('image.jpg', cv2.IMREAD_COLOR)

#提取RGB矩陣
(B, G, R) = cv2.split(img)

#取平均值，轉灰階
mean =  B/3 + G/3 + R/3
mean = np.uint8(mean)
gray_img = cv2.merge([mean, mean, mean])

# 顯示原圖
cv2.imshow('My Image', img)

# 按下任意鍵則關閉所有視窗
cv2.waitKey(0)
cv2.destroyAllWindows()

# 顯示灰階影像
cv2.imshow('My Image', gray_img)

# 按下任意鍵則關閉所有視窗
cv2.waitKey(0)
cv2.destroyAllWindows()

#儲存影像
cv2.imwrite('Original.jpg', img)
cv2.imwrite('Gray.jpg', gray_img)
