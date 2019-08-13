# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 19:48:56 2019

@author: cfd_Liu
"""
import cv2
import numpy as np
import face_recognition as fr

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
def draw_delaunay(img, subdiv, delaunay_color ) :
 
    triangleList = subdiv.getTriangleList();
    size = img.shape
    r = (0, 0, size[1], size[0])
 
    for t in triangleList :
         
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
         
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
         
            cv2.line(img, pt1, pt2, delaunay_color, 1)
            cv2.line(img, pt2, pt3, delaunay_color, 1)
            cv2.line(img, pt3, pt1, delaunay_color, 1)

if __name__ == '__main__':
    img = cv2.imread("./Clinton.jpg");
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    top, right, bottom, left = fr.face_locations(img)[0]
    img = img[top:bottom, left:right]
    #
    img_orig = img.copy();
    
    landmarks = fr.face_landmarks(img)#
    pointsTotal = []
    for t in landmarks[0].values():
        for point in t:
            pointsTotal.append(point)
            cv2.circle(img_orig, point, 2, (255,0,0), -1)
    pointsTotal = np.array(pointsTotal)
    hullIndex = cv2.convexHull(pointsTotal, returnPoints=False)
    points = pointsTotal[hullIndex[:,0]]
    
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR)
    cv2.imshow('', img_orig)
    cv2.waitKey(0)
    
    size = img.shape
    rect = (0, 0, size[1], size[0])
    
    subdiv = cv2.Subdiv2D(rect);
    
    for x,y in points :
        subdiv.insert((x,y))
    
    img_copy = img_orig.copy()
    draw_delaunay( img_copy, subdiv, (255, 255, 255) );
    cv2.imshow('', img_copy)
    cv2.waitKey(0)
    subdiv.getVertex()
