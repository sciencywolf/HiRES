#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import cv2
import string
from skimage.transform import pyramid_reduce
import sympy as sym
from scipy.ndimage import interpolation


def image_mode(im):
    hist = cv2.calcHist([im],[0],None,[256],[0,256])[:,0]
    mean = int(im.mean())
    light = hist[mean:].sum()
    dark = hist[:mean].sum()
    ratio = light/(light+dark)
    mode = ratio < 0.50
    if mode:
        im = cv2.bitwise_not(im)
    return im


def noise_removal(im, blur=None, radius=3):
    if not blur in (None, 'gaussian', 'median'):
        raise ValueError("blur must be None, 'gaussian' or 'median'")
    if blur=='gaussian':
        im = cv2.GaussianBlur(im, (radius,radius), 0)
    elif blur=='median':
        im = cv2.medianBlur(im, radius)
    return im


def residue_removal(im, areas=[1]):
    num_stats, labels, stats, centroids = cv2.connectedComponentsWithStats(cv2.bitwise_not(im), 8, cv2.CV_16U)
    im_clean = im.copy()
    for label in range(num_stats):
        if stats[label,cv2.CC_STAT_AREA] in areas:
            im_clean[labels == label] = 255
    return im_clean


def get_square(image, square_size):
    height, width = image.shape    
    if(height > width):
      differ = height
    else:
      differ = width
    differ += 2

    # square filler
    mask = np.zeros((differ, differ), dtype = "uint8")

    x_pos = int((differ - width) / 2)
    y_pos = int((differ - height) / 2)

    # center image inside the square
    mask[y_pos: y_pos + height, x_pos: x_pos + width] = image[0: height, 0: width]

    # downscale if needed
    if differ / square_size > 1:
      mask = (255*pyramid_reduce(mask, differ / square_size, multichannel=False)).astype(np.uint8)
    else:
      mask = cv2.resize(mask, (square_size, square_size), interpolation = cv2.INTER_AREA)
    return mask


def contours_extraction(im, crop=False, plot=False):
    im_not = np.bitwise_not(im)
    _, contours, hierarchy = cv2.findContours(im_not,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 

    markers = np.zeros(im.shape,dtype = np.int32)-1
    for id in range(len(contours)):
        cv2.drawContours(markers,contours,id,id, -1)

    bboxes = {}
    im_alpha = np.dstack((im, im, im))
    im_beta = im_alpha.copy()
    for index, contour in enumerate(contours):
        [x,y,w,h] = cv2.boundingRect(contour)
        marker = (im_not*(markers==index))[y:y+h, x:x+w]
        bboxes[index] = {'contour':contour, 'marker':marker,
                         'box':(x,y,w,h), 'centerx':x+0.5*w,
                         'centery':y+0.5*h, 'top':y+h,
                         'bottom':y, 'right':x+w,
                         'left':x}

        if w <1 and h<1:
            continue
        cv2.rectangle(im_alpha, (x,y),(x+w,y+h), (12,250,12),2)
        if crop == True:
            im_crop = im_alpha[y:y+h, x:x+w]
            s = '../output/crop_' + str(index) + '.jpg'
            cv2.imwrite(s , im_crop)
    
    alpha = 0.8
    beta = 1-alpha
    im_recta = cv2.addWeighted(im_alpha, alpha, im_beta, beta, 0.0)
    if plot==True:
        fig, ax = plt.subplots()
        ax.imshow(im_recta)
        fig.show()

    return bboxes


def neighbours(im):
    return im[2:,1:-1], im[2:,2:], im[1:-1,2:], im[:-2,2:], im[:-2,1:-1], im[:-2,:-2], im[1:-1,:-2], im[2:,:-2]


def transitions(P):
    return np.sum([((P[(i+1)%len(P)]-P[i])>0).astype(int) for i in range(len(P))], axis=0)


def zhangSuen(im):
    im = (im/255).astype(np.int)
    diff = np.zeros(im.shape)
    while not np.all(diff):
        diff = im.copy()

        P = neighbours(im)
        condition0 = im[1:-1,1:-1]
        condition4 = P[2]*P[4]*P[6]
        condition3 = P[0]*P[2]*P[4]
        condition2 = transitions(P) == 1
        condition1 = (2 <= np.sum(P,axis=0)) * (np.sum(P,axis=0) <= 6)
        cond = (condition0 == 1) * (condition4 == 0) * (condition3 == 0) * (condition2 == 1) * (condition1 == 1)
        changing1 = np.where(cond == 1)
        im[changing1[0]+1,changing1[1]+1] = 0

        P = neighbours(im)
        condition0 = im[1:-1,1:-1]
        condition4 = P[0]*P[4]*P[6]
        condition3 = P[0]*P[2]*P[6]
        condition2 = transitions(P) == 1
        condition1 = (2 <= np.sum(P,axis=0)) * (np.sum(P,axis=0) <= 6)
        cond = (condition0 == 1) * (condition4 == 0) * (condition3 == 0) * (condition2 == 1) * (condition1 == 1)
        changing2 = np.where(cond == 1)
        im[changing2[0]+1,changing2[1]+1] = 0
        diff = diff==im
    return im


def extract_glyphs(bboxes, plot=False):
    for i in range(len(bboxes.keys())):
        resized = get_square(bboxes[i]['marker'], 32)
        cv2.normalize(resized, resized, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        ret, res_norm = cv2.threshold(resized,125,255,cv2.THRESH_BINARY)
        skel = zhangSuen(res_norm).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))
        glyph = cv2.dilate(skel, kernel)
        if (i < 20) & plot:
            fig, ax = plt.subplots()
            ax.imshow(glyph, cmap=cm.binary)
            fig.show()


def show_outlines(bboxes):
    fig, ax = plt.subplots()
    for i in range(len(bboxes.keys())):
        x,y = np.hsplit(bboxes[i]['contour'].reshape(-1,2),2)
        ax.plot(x, y)
    ax.invert_yaxis()
    ax.set_aspect('equal')


def deskew(im):
    d_angle = 10
    c_angle = 0
    for delta in [10,1,0.1]:
        angles = np.arange(c_angle-d_angle, c_angle+d_angle, delta)
        scores = []
        for angle in angles:
            data = interpolation.rotate(im, angle, reshape=False, order=0, cval=255)
            #hist = np.sum(data, axis=1)
            hist = cv2.reduce(data, 1, cv2.REDUCE_AVG)
            score = np.std(hist)**2
            scores.append(score)
        best_score = max(scores)
        c_angle = best_angle = angles[scores.index(best_score)]
        d_angle = delta
    im_rot = interpolation.rotate(im, best_angle, reshape=False, order=0, cval=255)
    return im_rot

def segment_lines(im):
    im_proj = cv2.reduce(im, 1, cv2.REDUCE_AVG)
    hist = im_proj == 255
    ycoords = []
    y = 0
    count = 0
    isSpace = False
    for i in range(len(hist)):
        if not isSpace:
            if hist[i]:
                isSpace = True
                count = 1
                y = i
        else:
            if not hist[i]:
                isSpace = False
                ycoords.append(y/count)
            else:
                y += i
                count += 1
    lines = [line for line in np.vsplit(im, np.array(ycoords).astype(int)) if not (line==255).all()]
    
    return lines

def split_formula(fpath, blur=None, radius=3, areas=[1], crop=False, plot=False):
    im = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (im.shape[1]*2, im.shape[0]*2), interpolation = cv2.INTER_CUBIC)
    cv2.normalize(im, im, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    im = image_mode(im)

    im_blur = noise_removal(im, blur, radius)
    im_thresh = cv2.adaptiveThreshold(im_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 20)

    im_clean = residue_removal(im_thresh, areas)
    im_deskew = deskew(im_clean)
    #lines = segment_lines(im_deskew)
    lines = [im_deskew]

    for line in lines:
        bboxes = contours_extraction(line, crop, plot)
        extract_glyphs(bboxes, plot=True)
        #plt.plot(np.ones((line.shape[1]))*np.median([bboxes[i]['centery'] for i in bboxes]))


if __name__ == "__main__":
    fpath = '../datasets/formulas/im2.png'
    split_formula(fpath, areas=range(25), plot=True)
