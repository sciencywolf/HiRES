#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.io import imread

def show_glyph_sample(latex, dspath, nb=6):
    """
    ======
    show_glyphs_sample
    ======

    Definition: show_glyph_sample(latex, ds)
    Type: Function

    ----

    Show image sample corresponding to a latex command for given dataset(s)

    Parameters
    ----------
    latex : string
        a latex command corresponding to a symbol classe in dataset
    dspath: string or list of string
        path of the dataset(s) to look into.
    """
    if (nb%2!=0): nb+=1
    m = np.max([i if (nb%i==0) else 0 for i in range(1,np.int(np.sqrt(nb))+1)])
    n = nb//m
    
    if type(dspath)==list:
        dfs = [pd.read_csv(path) for path in dspath]
        subdfs = [df[df['latex']==latex]['path'] for df in dfs]
        
        for subdf, path in zip(subdfs, dspath):
            fig, ax = plt.subplots(m,n)
            for i,k in enumerate(np.random.randint(0, len(subdf), nb)):
                im = imread(os.path.dirname(path)+'/'+subdf.iloc[k])
                if len(im.shape)==3:
                    im = im[:,:,0]
                ax[i//n,i%n].imshow(im, cmap=cm.binary_r)
    else:
        df = pd.read_csv(dspath)
        subdf = df[df['latex']==latex]['path']

        fig, ax = plt.subplots(m,n)
        for i,k in enumerate(np.random.randint(0, len(subdf), nb)):
            im = imread(os.path.dirname(dspath)+'/'+subdf.iloc[k])
            if len(im.shape)==3:
                    im = im[:,:,0]
            ax[i//n,i%n].imshow(im, cmap=cm.binary_r)
