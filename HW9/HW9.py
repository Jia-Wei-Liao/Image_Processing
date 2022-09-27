# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import math

def his(img, s):
    [img_his, img_bin] = np.histogram(img.flatten(), range(257))
    plt.bar(range(256), img_his, color = 'blue')
    plt.savefig(s + '.png')

def imgshow(img):
    cv2.imshow('My Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

n, m = 256, 256
g = (np.ones((n, m))*100).astype('uint8')
his(g, 'g_histogram')
imgshow(g)

mu = 0
sigma = 15**(1/2)
f = (np.zeros((n, m))).astype('uint8')

for i in range(n):
    for j in range(m):
        if j%2 == 0:
            r, phi = random.random(), random.random()
            z1 = sigma*math.cos(2*math.pi*phi)*((-2)*math.log(r))**(1/2)
            z2 = sigma*math.sin(2*math.pi*phi)*((-2)*math.log(r))**(1/2)
            f[i, j] = g[i, j] + z1
            f[i, j+1] = g[i, j+1] + z2

f1 = (np.zeros((n, m))).astype('uint8')
for i in range(n):
    for j in range(m):
        if f[i, j] == 0:
            f1[i, j] == 0
        elif f[i, j] > 256-1:
            f1[i, j] = 256-1
        else:
            f1[i, j] = f[i, j]

his(f1, 'f1_histogram')
imgshow(f1)

f2 = (np.zeros((n, m))).astype('uint8')
for i in range(n):
    for j in range(m):
        s = np.random.normal(mu, sigma) #Create Gaussian Noise
        f2[i, j] = g[i, j] + s
his(f2, 'f2_histogram')
imgshow(f2)
