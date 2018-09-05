#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
import sympy as sym
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import threading
import multiprocessing
import string
import cv2
import sympy as sym
from formula import contours_extraction

#%% Generation of hierarchical data
def append(x,y):
    return("{} {}".format(x,y))

def frac(x,y):
    return('\\frac{{{}}}{{{}}}'.format(x,y))

def sub(x,y):
    return('{}_{{{}}}'.format(x,y))

def sup(x,y):
    return('{}^{{{}}}'.format(x,y))

def ope(o,x,y):
    return('{}_{{{}}}^{{{}}}'.format(o,x,y))

def sqrt(x):
    return('\\sqrt{{{}}}'.format(x))

def render(s, fname='file', show_only=True):
    if show_only:
        sym.preview(r'$${}$$'.format(s), output='png', viewer='gloobus-preview', euler=False, dvioptions=["-D","300","-T","tight"])
    else:
        sym.preview(r'$${}$$'.format(s), viewer='file', filename='{}.png'.format(fname), euler=False, dvioptions=["-D","300","-T","tight"])

def combine(func, *args):
    return(func(*args))
    
def generate(generator, size=10):
    elems = []
    for label, method, args in generator:
        argmods = [np.random.choice(args[k], size=size) for k in args.keys()]
        for f_args in zip(*argmods):
            elems.append([combine(method, *f_args), label])
    return(elems)

ll = [s for s in string.ascii_lowercase]
lu = [s for s in string.ascii_uppercase]
ld = [s for s in string.digits]
lp = ['(',')','[',']','\{','\}']
lg = ['\\alpha','\\beta','\\gamma','\\delta',
      '\\epsilon','\\zeta','\\eta','\\theta',
      '\\iota','\\kappa','\\lambda','\\mu',
      '\\nu','\\xi','\\omicron','\\pi',
      '\\rho','\\sigma','\\tau','\\upsilon',
      '\\phi','\\chi','\\psi','\\omega']
lbo= ['\\sum','\\prod','\\int']
lo = ['+','-','\\times','/']
le = ['=','\\neq','>','<','\\leq','\\geq','\\sim','\\propto','\\equiv']

symL = ll+lu+ld+lp+lg+lbo+lo
symS = ll+lu+ld+lg

generator = [('right', append, {'0':symL,
                                '1':symL}),
             ('top', frac, {'0':symS,
                            '1':['']}),
             ('bottom', frac, {'0':[''],
                               '1':symS}),
             ('subscript', sub, {'0':symS,
                                 '1':symS}),
             ('superscript', sup, {'0':symS,
                                   '1':symS}),
             ('op_top', ope, {'0':lbo,
                              '1':symS,
                              '2':['']}),
             ('op_bottom', ope, {'0':lbo,
                                 '1':[''],
                                 '2':symS}),
             ('inside', sqrt, {'0':symS})]


randBefore = np.random.uniform(0,2,3000)
randAfter = np.random.uniform(0,2,3000)
symb = np.random.choice(symL, 3000)
symSpaces = ['\\hspace{{{}cm}} {} \\hspace{{{}cm}}'.format(b,x,a) for b,x,a in zip(randBefore,symb,randAfter)]

generator_space = [('top', frac, {'0':symSpaces,
                                  '1':['']}),
                   ('bottom', frac, {'0':[''],
                                     '1':symSpaces}),
                   ('inside', sqrt, {'0':symSpaces})]

def generate_samples(generator, size=1000):
    a = generate(generator, size)
    impath = '../datasets/Formulas/hierarchy-data-spaces/'
    pool = multiprocessing.Pool(processes=3)
    for i, eq in enumerate(a):
        pool.apply_async(render, (eq[0],), {'fname':'{}{}-{}'.format(impath,eq[1],str(i)),'show_only':False})
    pool.close()
    pool.join()


#%% Features extraction from hierarchical data
def extract_features(bboxA, bboxB):
    keys = ['left', 'right', 'top', 'bottom','centerx', 'centery']
    [lA, rA, tA, bA, cxA, cyA] = [bboxA[k] for k in keys]
    [lB, rB, tB, bB, cxB, cyB] = [bboxB[k] for k in keys]
    scalerW = 1./(max(rA,rB) - min(lA,lB)) # scale factor using global bounding box width
    scalerH = 1./(max(tA,tB) - min(bA,bB)) #scale factor using global bounding box height
    features = [(lB-lA)*scalerW, (rB-rA)*scalerW,
                (tB-tA)*scalerH, (bB-bA)*scalerH,
                (cxB-cxA)*scalerW, (cyB-cyA)*scalerH,
                np.arctan2(cyB-cyA, cxB-cxA),
                (lB-cxA)*scalerW, (rB-cxA)*scalerW,
                (tB-cyA)*scalerH, (bB-cyA)*scalerH,
                np.arctan2(tB-bA, (rB-lB)-rA), 
                np.arctan2(bB-bA, (rB-lB)-rA),
                np.arctan2(tB-tA, (rB-lB)-rA), 
                np.arctan2(bB-tA, (rB-lB)-rA)]
    return features

path = '../datasets/Formulas/hierarchy-data/'

def generate_pair_dataset(path, dsname):
    data = []
    for fname in os.listdir(path):
        fpath = path+fname
        im = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        im_res = cv2.resize(im, (im.shape[1]*2, im.shape[0]*2), interpolation = cv2.INTER_CUBIC)
        ret, im_bin = cv2.threshold(im_res,125,255,cv2.THRESH_BINARY)
        
        bboxes = contours_extraction(im_bin)
        
        relation = fname.split('-')[0]
        if len(bboxes)!=2:
            warning = True
        else:
            warning = False
        features = extract_features(bboxes[0], bboxes[1])
        features += [relation, fpath, warning]
        data.append(features)
    
    labels = ['dist_left','dist_right',
              'dist_top','dist_bottom',
              'dist_cx', 'dist_cy', 'angle', 
              'dist_cx_left', 'dist_cx_right', 
              'dist_cy_top', 'dist_cy_bottom', 
              'dist_bottomright_middletop', 'dist_bottomright_middlebottom', 
              'dist_topright_middletop', 'dist_topright_middlebottom',
              'relationship', 'path','warning']
    df = pd.DataFrame.from_records(data, columns=labels)
    if dsname not in os.listdir('../datasets/'):
        df.to_csv('../datasets/'+dsname, index=False)
    else:
        print('File aleady exists')


#%% Classifiation
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

#df = pd.read_csv('../datasets/relation_20180905.csv')
df1 = pd.read_csv('../datasets/relation_20180905.csv')
df2 = pd.read_csv('../datasets/relation_20180905_spaces.csv')
df = pd.concat([df1, df2], ignore_index=True)
labels = ['dist_left','dist_right',
          'dist_top','dist_bottom',
          'dist_cx', 'dist_cy', 'angle', 
          'dist_cx_left', 'dist_cx_right', 
          'dist_cy_top', 'dist_cy_bottom',
          'dist_bottomright_middletop', 'dist_bottomright_middlebottom', 
          'dist_topright_middletop', 'dist_topright_middlebottom'] 
#labels = ['dist_left','dist_right','dist_top','dist_bottom','dist_cx', 'dist_cy', 'angle']
target = df[df.warning==False]["relationship"]
data = df[df.warning==False][labels]

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, shuffle=True, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm = SVC()

param_grid = {
    'kernel': ['linear', 'rbf'],
    'C': [1.0, 2., 3., 5., 8., 10.],
    'gamma': ['auto', 0.1, 1.0, 1.5, 2]
}

cv_svm = GridSearchCV(estimator=svm, scoring='f1_micro', param_grid=param_grid, cv= 5)

cv_svm.fit(X_train_scaled, y_train)

test_pred = cv_svm.predict(X_test_scaled)
train_pred = cv_svm.predict(X_train_scaled)

print('score train',cv_svm.score(X_train_scaled, y_train))
print('score test',cv_svm.score(X_test_scaled, y_test))
print(classification_report(y_train, train_pred, digits=3))
print(classification_report(y_test, test_pred, digits=3))


