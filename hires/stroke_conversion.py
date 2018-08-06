#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def stroke(elem):
    """
    ======
    stroke
    ======

    Definition: stroke(elem)
    Type: Function

    ----

    Take an element from a dataframe and return the list of strokes

    Parameters
    ----------
    elem : pandas Series
        element of a dataframe containing a stroke variable

    Returns
    -------
    strokes : list of arrays
        List of arrays containing stroke coordinates.
    """
    strokes_raw = eval(elem['strokes'])
    strokes = [np.array(strokes_raw[j])[:, :2] for j in range(len(strokes_raw))]
    return strokes


def draw(strokes):
    """
    ======
    draw
    ======

    Definition: draw(strokes)
    Type: Function

    ----

    Plot strokes

    Parameters
    ----------
    strokes : list of arrays
        List of arrays containing stroke coordinates
    """
    fig, ax = plt.subplots()
    for s in strokes:
        ax.plot(s[:, 0], s[:, 1])
    ax.invert_yaxis()
    plt.show()


def display(glyph):
    """
    ======
    display
    ======

    Definition: display(glyph)
    Type: Function

    ----

    Show the symbole image

    Parameters
    ----------
    glyph : numpy array
        array of an image
    """
    fig, ax = plt.subplots()
    ax.imshow(glyph, interpolation='none',
              origin='upper',
              cmap=plt.cm.gray,
              vmin=0,
              vmax=1)
    ax.grid(which='major')
    ax.set_xticks(np.arange(0, 32)+0.5)
    ax.set_yticks(np.arange(0, 32)+0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.show()


def xiaoline(p1, p2):
    """
    Uses Xiaolin Wu's line algorithm to interpolate all of the pixels along
    a straight line, given two points (x1, y1) and (x2, y2)

    This function is based on pseudo code provided on wikipedia article:
    http://en.wikipedia.org/wiki/Xiaolin_Wu's_line_algorithm

    ======
    xiaoline
    ======

    Definition: xiaoline(x1, y1, x2, y2)
    Type: Function

    ----

    Generates coordinates of pixels forming the segment between two points p1 (x1,y1) and
    p2 (x2,y2).

    Parameters
    ----------
    x1, y1, x2, y2 : numbers
        coordinates of points p1 and p2

    Returns
    -------
    coords : list of tuples
        List of coordinates for each pixel belonging to the segment (p1,p2).
    """

    x = []
    y = []
    x1, y1 = p1
    x2, y2 = p2

    dx = x2-x1
    dy = y2-y1
    steep = abs(dy) > abs(dx)

    if steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
        dy, dx = dx, dy

    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1

    if dx != 0.:
        gradient = dy / dx
    else:
        gradient = 1.

    # handle first endpoint
    xend = round(x1)
    yend = y1 + gradient * (xend - x1)
    xpxl1 = int(xend)
    ypxl1 = int(yend)
    x.append(xpxl1)
    y.append(ypxl1)
    x.append(xpxl1)
    y.append(ypxl1+1)
    intery = yend + gradient

    # handles the second point
    xend = round(x2)
    yend = y2 + gradient * (xend - x2)
    xpxl2 = int(xend)
    ypxl2 = int(yend)
    x.append(xpxl2)
    y.append(ypxl2)
    x.append(xpxl2)
    y.append(ypxl2 + 1)

    # main loop
    for px in range(xpxl1 + 1, xpxl2):
        x.append(px)
        y.append(int(intery))
        x.append(px)
        y.append(int(intery) + 1)
        intery = intery + gradient

    if steep:
        y, x = x, y

    coords = list(zip(x, y))

    return coords


def stroke2image(strokes, size=32, margin=0, invert=False):
    """
    ======
    stroke2image
    ======

    Definition: stroke2image(strokes, size=32, margin=0, invert=False)
    Type: Function

    ----

    Generate an array containing pixels values from strokes coordinates

    Parameters
    ----------
    strokes : list of array
        List of array containing strokes coordinates
    size : number
        Size in pixel of the image
    margin : number
        Wether to add a margin around the symbol
    invert : boolean
        Choose between white over black if True or black over white if False

    Returns
    -------
    glyph : array
        Image of the symbol
    """
    # Determine strokes boundaries
    extval = np.array([[s.min(axis=0), s.max(axis=0)] for s in strokes]).reshape((-1, 2))
    xmin, ymin, xmax, ymax = np.array([f(extval, 0) for f in [np.min, np.max]]).flatten()

    # Translate to origin and rescale largest dimension to 1
    if (xmin == xmax) & (ymin == ymax):
        strokes_norm = [s-[xmin, ymin] for s in strokes]
    else:
        strokes_norm = [(s-[xmin, ymin])/(max([xmax-xmin, ymax-ymin])) for s in strokes]
    strokes_scaled = [margin+s*((size-2)-2*margin) for s in strokes_norm]

    glyph = np.zeros((size, size))
    pxcoords = []
    for s in strokes_scaled:
        for p1, p2 in zip(s[:-1], s[1:]):
            coords = xiaoline(p1, p2)
            pxcoords += coords
    for px in pxcoords:
        glyph[px] = 1

    offset0 = next((round((size-i)/2) for i in range(size) if all((glyph.sum(0) == 0)[i:])), 0)
    offset1 = next((round((size-i)/2) for i in range(size) if all((glyph.sum(1) == 0)[i:])), 0)
    glyph = np.roll(np.roll(glyph, offset0, axis=1), offset1, axis=0).T

    if not invert:
        glyph = abs(glyph-1)
    glyph *= 255
    return glyph


def image_generator(d):
    """
    ======
    image_generator
    ======

    Definition: image_generator(d)
    Type: Function

    ----

    Generate images for every elements in the dataframe

    Parameters
    ----------
    d : dataframe
        Dataframe containing elements with strokes to draw
    """
    init = input("Initiate symbols image generation? (y/n): ")
    if init == 'y':
        for i in range(len(d)):
            elem = d.loc[i]
            print(i, elem['key'])
            strokes = stroke(elem)
            glyph = stroke2image(strokes, size=32, margin=0)
            image = Image.fromarray(glyph.astype(np.uint8))
            image.save('../datasets/Detexify/{:06.0f}.png'.format(i))


df = pd.read_csv('../datasets/Detexify/detexify.csv')
image_generator(df)
