# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 15:00:03 2019

@author: cfd_Liu
"""

import cv2
import numpy as np
import face_recognition as fr
def showTri(imgA, imgB, triListA,triListB):
    for x1,y1,x2,y2,x3,y3 in triListA:
        cv2.line(imgA, (x1,y1), (x2,y2), (255,0,0), 1)
        cv2.line(imgA, (x2,y2), (x3,y3), (255,0,0), 1)
        cv2.line(imgA, (x3,y3), (x1,y1), (255,0,0), 1)
    for x1,y1,x2,y2,x3,y3 in triListB:
        cv2.line(imgB, (x1,y1), (x2,y2), (255,0,0), 1)
        cv2.line(imgB, (x2,y2), (x3,y3), (255,0,0), 1)
        cv2.line(imgB, (x3,y3), (x1,y1), (255,0,0), 1)
    cv2.imshow('Clinton', imgA)
    cv2.imwrite('./imgs/1.jpg', imgA, [cv2.IMWRITE_JPEG_QUALITY, 100])
    cv2.imshow('Trump', imgB)
    cv2.imwrite('./imgs/2.jpg', imgB, [cv2.IMWRITE_JPEG_QUALITY, 100])
    
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True
def delaunay(r, subdiv, points, pointsIndex) :
    triangleList = subdiv.getTriangleList();
    List = []
    triIndList = []
    for t in triangleList :
        IndList = []
        pt = np.reshape(t, [3,2]).astype(np.int32)

        if rect_contains(r, pt[0]) and rect_contains(r, pt[1]) and rect_contains(r, pt[2]) :
            List.append(True)
            for point in pt:
                for i, t in enumerate(points):
                    if point[0]==t[0] and point[1]==t[1]:
                        IndList.append(pointsIndex[i])
            triIndList.append(IndList)
        else:
            List.append(False)
    return triangleList[List].astype(np.int32), np.array(triIndList)
def caltriListB(pointsTotal, triIndList):
    triList = np.zeros([triIndList.shape[0], 6], np.int32)
    for num,index in enumerate(triIndList):
        triList[num, 0:2] = pointsTotal[index[0]]
        triList[num, 2:4] = pointsTotal[index[1]]
        triList[num, 4:6] = pointsTotal[index[2]]
    return triList
imgA = cv2.imread('./imgs/Clinton.jpg', 1)

imgRGBA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
landmarksA = fr.face_landmarks(imgRGBA)
pointsTotalA = []
for t in landmarksA[0].values():
    for point in t:
        pointsTotalA.append(point)
pointsTotalA = np.array(pointsTotalA)
pointsIndexA = cv2.convexHull(pointsTotalA, returnPoints = False)[:,0]
pointsA = pointsTotalA[pointsIndexA, :]
#for x,y in pointsA:
#    cv2.circle(imgA, (x,y), 2, (0,0,255))
#    cv2.imshow('',imgA)
#    cv2.waitKey(0)

imgB = cv2.imread('./imgs/Trump.png', 1)

imgRGBB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)
landmarksB = fr.face_landmarks(imgRGBB)
pointsTotalB = []
for t in landmarksB[0].values():
    for point in t:
        pointsTotalB.append(point)
pointsTotalB = np.array(pointsTotalB)
pointsB = pointsTotalB[pointsIndexA, :]
#for x,y in pointsB:
#    cv2.circle(imgB, (x,y), 2, (0,0,255))
#    cv2.imshow('',imgB)
#    cv2.waitKey(0)
    
#rectB = (0,0,imgB.shape[1], imgB.shape[0])
#subdiv = cv2.Subdiv2D(rectB)
#for x,y in pointsB:
#    subdiv.insert((x,y))
#triListB = delaunay(rectB, subdiv)

rectA = (0,0,imgA.shape[1], imgA.shape[0])
subdiv = cv2.Subdiv2D(rectA)
for x,y in pointsA:
    subdiv.insert((x,y))
triListA, triIndList = delaunay(rectA, subdiv, pointsA, pointsIndexA)
triListB = caltriListB(pointsTotalB, triIndList)

showTri(imgA.copy(), imgB.copy(), triListA,triListB)

imgSrc = imgB.copy()
for i, A in enumerate(triListA):
    mask = np.zeros(imgB.shape, imgB.dtype)
    matA = np.reshape(A, [3,2]) 
    matB = np.reshape(triListB[i], [3,2])
    matAffine = cv2.getAffineTransform(matA.astype(np.float32), matB.astype(np.float32)) #mat>>float32
    img = cv2.warpAffine(imgA, matAffine, (imgB.shape[1], imgB.shape[0]))
    cv2.fillConvexPoly(mask, matB, (1,1,1))
    img = img * mask
    cv2.fillConvexPoly(imgSrc, matB, (0,0,0))
    imgSrc = imgSrc + img
cv2.imshow('imgSrc', imgSrc)
cv2.imwrite('./imgs/3.jpg', imgSrc, [cv2.IMWRITE_JPEG_QUALITY, 100])

mask = np.zeros(imgB.shape, imgB.dtype)
poly = pointsB
cv2.fillConvexPoly(mask, poly, (255,255,255))
x2,y2 = np.max(pointsB, axis=0)
x1,y1 = np.min(pointsB, axis=0)
center = ((x1+x2)//2, (y1+y2)//2)
cv2.imshow('mask', mask)
cv2.imwrite('./imgs/4.jpg', mask, [cv2.IMWRITE_JPEG_QUALITY, 100])

output = cv2.seamlessClone(imgSrc, imgB, mask, center, cv2.NORMAL_CLONE)
cv2.imshow('out', output)
cv2.imwrite('./imgs/5.jpg', output, [cv2.IMWRITE_JPEG_QUALITY, 100])
#
#imgc = imgB.copy()
#cv2.fillConvexPoly(imgc, np.reshape(triListB[0], [-1, 2]).astype(np.int32), (0,0,255))
#cv2.imshow('', imgc)
#cv2.waitKey(0)
