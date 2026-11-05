import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("12345.png",cv.IMREAD_GRAYSCALE)
cv.imshow('input img_a',img)
height, width = img.shape

img_pad = np.zeros((height+2, width+2), np.uint8)

img_pad_d = np.zeros((height+2, width+2), np.uint8)



for y in range(1,height+1):
    for x in range(1,width+1):
        img_pad[y,x] = img[y-1,x-1]

##opening

kernel = np.ones((3, 3), np.uint8)
kh,kw = kernel.shape

flags = np.zeros((height+1, width+1), np.uint8)
for y in range(1,height+1):
    for x in range(1,width+1):
        flag = img_pad[y-1,x-1]
        for yk in range (-1,2):
            for xk in range (-1,2):
                if flag > img_pad[y+yk,x+xk] :
                    flag = img_pad[y+yk,x+xk]
        flags[y,x] = flag

for y in range(height +1):
    for x in range(width +1):
        img_pad[y,x] = flags[y,x]


flags = np.zeros((height+1, width+1), np.uint8)
for y in range(1,height+1):
    for x in range(1,width+1):
        flag = img_pad[y-1,x-1]
        for yk in range (-1,2):
            for xk in range (-1,2):
                if flag < img_pad[y+yk,x+xk] :
                    flag = img_pad[y+yk,x+xk]
        flags[y,x] = flag

for y in range(height +1):
    for x in range(width +1):
        img_pad_d[y,x] = flags[y,x]


img_res = np.zeros((height, width), np.uint8)
for y in range(height):
    for x in range(width):
        img_res[y,x]= img_pad_d[y+1,x+1]

cv.imshow('a_opening',img_res)


##closing

img2 = cv.imread("closer.png",cv.IMREAD_GRAYSCALE)
cv.imshow('input img_b',img2)
height, width = img2.shape

img2_pad = np.zeros((height+2, width+2), np.uint8)

for y in range(1,height+1):
    for x in range(1,width+1):
        img2_pad[y,x] = img2[y-1,x-1]

kernel = np.ones((3, 3), np.uint8)
flags = np.zeros((height+1, width+1), np.uint8)
for y in range(1,height+1):
    for x in range(1,width+1):
        flag = img2_pad[y-1,x-1]
        for yk in range (-1,2):
            for xk in range (-1,2):
                if flag < img2_pad[y+yk,x+xk] :
                    flag = img2_pad[y+yk,x+xk]
        flags[y,x] = flag

for y in range(height +1):
    for x in range(width +1):
        img2_pad[y,x] = flags[y,x]

kernel = np.ones((3, 3), np.uint8)
flags = np.zeros((height+1, width+1), np.uint8)
for y in range(1,height+1):
    for x in range(1,width+1):
        flag = img2_pad[y-1,x-1]
        for yk in range (-1,2):
            for xk in range (-1,2):
                if flag > img2_pad[y+yk,x+xk] :
                    flag = img2_pad[y+yk,x+xk]
        flags[y,x] = flag

for y in range(height +1):
    for x in range(width +1):
        img2_pad[y,x] = flags[y,x]


img2_res = np.zeros((height, width), np.uint8)
for y in range(height):
    for x in range(width):
        img2_res[y,x]= img2_pad[y+1,x+1]

cv.imshow('b_closing',img2_res)


##dilation

img3 = cv.imread("12345.png",cv.IMREAD_GRAYSCALE)
height, width = img3.shape

img3_pad = np.zeros((height+2, width+2), np.uint8)

for y in range(1,height+1):
    for x in range(1,width+1):
        img3_pad[y,x] = img3[y-1,x-1]

kernel = np.array([[0,1,0],
                   [1,1,1],
                   [0,1,0]], np.uint8)
flags = np.zeros((height+1, width+1), np.uint8)
for y in range(1,height+1):
    for x in range(1,width+1):
        flag = img3_pad[y,x]
        for yk in range (-1,2):
            for xk in range (-1,2):
                if kernel[yk+1,xk+1] == 1 :
                    if flag < img3_pad[y+yk,x+xk] :
                        flag = img3_pad[y+yk,x+xk]
        flags[y,x] = flag

for y in range(height +1):
    for x in range(width +1):
        img3_pad[y,x] = flags[y,x]

img3_res = np.zeros((height, width), np.uint8)
for y in range(height):
    for x in range(width):
        img3_res[y,x]= img3_pad[y+1,x+1]

cv.imshow('a_dilation',img3_res)


##erosion

img4 = cv.imread("12345.png",cv.IMREAD_GRAYSCALE)
height, width = img4.shape

img4_pad = np.zeros((height+2, width+2), np.uint8)

for y in range(1,height+1):
    for x in range(1,width+1):
        img4_pad[y,x] = img4[y-1,x-1]

kernel = np.array([[0,1,0],
                   [1,1,1],
                   [0,1,0]], np.uint8)
flags = np.zeros((height+1, width+1), np.uint8)
for y in range(1,height+1):
    for x in range(1,width+1):
        flag = img4_pad[y,x]
        for yk in range (-1,2):
            for xk in range (-1,2):
                if kernel[yk+1,xk+1] == 1 :
                    if flag > img4_pad[y+yk,x+xk] :
                        flag = img4_pad[y+yk,x+xk]
        flags[y,x] = flag

for y in range(height +1):
    for x in range(width +1):
        img4_pad[y,x] = flags[y,x]

img4_res = np.zeros((height, width), np.uint8)
for y in range(height):
    for x in range(width):
        img4_res[y,x]= img4_pad[y+1,x+1]

cv.imshow('a_erosion',img4_res)
cv.waitKey(0)
