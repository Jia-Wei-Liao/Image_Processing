# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import cv2

def his(img, s):
    [img_his, img_bin] = np.histogram(img.flatten(), range(257))
    plt.bar(range(256), img_his, color = 'blue')
    plt.savefig(s + '_his.png')
    return img_his

def imgshow(img):
    cv2.imshow('My Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def a(t, p):
    return sum(p[:t+1])

def b(t, p):
    return sum(p[t+1:256])

def m(t, p):
    return sum(p[:256]*range(256))

def ma(t, p):
    return sum(p[:t+1]*range(t+1))

def val(t, p):
    return (ma(t, p)-m(t, p)*a(t, p))**2/(a(t, p)*b(t, p))

I = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
r, c = np.shape(I)
p = his(I, 'image')/(r*c)
maxval, t_index = max([(val(t, p), t) for t in range(50, 210)])
Out = np.zeros((r, c)).astype('uint8')
binary = lambda x:255 if x>t_index else 0

for i in range(r):
    for j in range(c):
        Out[i, j] = binary(I[i, j])
imgshow(Out)
cv2.imwrite('Otsus.jpg', Out)
