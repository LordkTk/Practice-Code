# -*- coding: utf-8 -*-
"""
Created on Mon May 13 18:34:31 2019

@author: cfd_Liu
"""

import cv2
import numpy as np

def Sobel(warped, low=20, high=255):
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    grad = cv2.Sobel(gray, cv2.CV_64F, 1,0, ksize=3)
    grad_abs = np.absolute(grad)
    grad_norm = np.uint8(grad_abs/np.max(grad_abs)*255)
    
    low = 20; high = 255
    dst = np.zeros_like(grad_norm)
    dst[(grad_norm>=low) & (grad_norm<=high)] = 1
    cv2.imshow('sobel',dst*255)
    return dst
def hsv(warped):
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    yellow_low = np.array([0,100,100])
    yellow_high = np.array([80,255,255])
    white_low = np.array([0,0,160])
    white_high = np.array([255,80,255])
    mask_yl = cv2.inRange(hsv, yellow_low, yellow_high)
    mask_wt = cv2.inRange(hsv, white_low, white_high)
    mask = cv2.bitwise_or(mask_yl, mask_wt)
    mask[mask>0] = 1
    cv2.imshow('y and w', mask*255)
    return mask
img = cv2.imread('./imgs/src.jpg', 1)
(H, W, _) = img.shape

warp_offset = 100
corners_src = [(73,H), (W-58,H), (267,170), (207,170)]
corners_dst = [(warp_offset,H), (W-warp_offset,H), (W-warp_offset,50), (warp_offset,50)]
M = cv2.getPerspectiveTransform(np.float32(corners_src), np.float32(corners_dst))
warped = cv2.warpPerspective(img, M, (W,H))
cv2.imshow('src', warped)
cv2.imwrite('./imgs/1.jpg', warped, [cv2.IMWRITE_JPEG_QUALITY, 100])

mask_sobel = Sobel(warped)
mask_y_w = hsv(warped)
#mask = mask_sobel | mask_y_w
mask = mask_y_w
cv2.imshow('mask', mask*255)
cv2.imwrite('./imgs/2.jpg', mask, [cv2.IMWRITE_JPEG_QUALITY, 100])

winNumx = 10
winNumy = 10
winSizex = W//winNumx
winSizey = H//winNumy

peak = np.sum(mask, axis=0)
rec = []

peak[peak<20] = 0
while np.max(peak)>0:
    mid = np.argmax(peak)
    xlow = mid-winSizex//2 if mid+winSizex//2<=W else W-winSizex
    xlow = xlow if xlow>0 else 0
    rec.append(xlow)
    peak[xlow:xlow+winSizex] = 0

pointsTotal = []
maskDst = np.dstack((mask, mask, mask))*255
for xlow in rec:
    points = []
    yhigh = H+1; ylow = H-winSizey+1
    for _ in range(winNumy):
        cv2.rectangle(maskDst, (xlow, ylow), (xlow+winSizex, yhigh), (0,255,0), 1)
        (y_inds, x_inds) = mask[ylow:yhigh,xlow:xlow+winSizex].nonzero()
        if len(y_inds)*len(x_inds)<20:
            midx = xlow + winSizex//2
            midy = ylow + winSizey//2
            xlow = xlow
        else:
            for i in range(len(x_inds)):
                maskDst[y_inds[i]+ylow,x_inds[i]+xlow] = [0,0,255]
            midx = np.int(np.mean(x_inds) + xlow)
            midy = np.int(np.mean(y_inds) + ylow)
            xlow = midx - winSizex//2
        
        points.append((midx,midy))
        yhigh = ylow
        ylow = ylow - winSizey
    pointsTotal.append(points)
for i in pointsTotal:
    for j in i:
        cv2.circle(warped, j, 2, (0,0,255))
cv2.imshow('1', warped)
cv2.imshow('maskDst', maskDst)

fitTotal = []
for points in pointsTotal:
    points = np.array(points)
    fit = np.polyfit(points[:,1], points[:,0], 2)
    fitTotal.append(fit)
plot = []
for fit in fitTotal:
    ply = np.linspace(0, H-1, H, dtype=np.int32)
    plx = fit[0]*ply**2 + fit[1]*ply + fit[2]
    pl = np.int32(np.transpose(np.vstack([plx, ply])))
    plot.append(pl)
    for x,y in pl:
        maskDst[y,x] = [255,255,255]
cv2.imshow('fitpoly', maskDst)
cv2.imwrite('./imgs/3.jpg', maskDst, [cv2.IMWRITE_JPEG_QUALITY, 100])

maskOut = np.zeros_like(maskDst)
cv2.fillPoly(maskOut, [np.vstack([plot[0], np.flipud(plot[1])])], (255,0,0))
M_rev = cv2.getPerspectiveTransform(np.float32(corners_dst), np.float32(corners_src))
out = cv2.warpPerspective(maskOut, M_rev, (W,H))
cv2.imshow('out', out)

Dst = cv2.addWeighted(img, 1, out, 0.5, 0)
cv2.imshow('Dst', Dst)
cv2.imwrite('./imgs/4.jpg', Dst, [cv2.IMWRITE_JPEG_QUALITY, 100])
