#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.chdir('/home/lucien/Documents/Formation-Data-Scientist/Projet/HiRES/hires/')

import time
import numpy as np
import pandas as pd
import sympy as sym
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import multiprocessing
import string
import cv2
import sympy as sym
from formula import contours_extraction, test_formula, extract_glyphs
import networkx as nx


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
        im = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)[::-1]
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

#generate_pair_dataset('../datasets/Formulas/hierarchy-data/', 'relation_20180912.csv')

#%% Classifiation
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

#df = pd.read_csv('../datasets/relation_20180905.csv')
df1 = pd.read_csv('../datasets/relation_20180912.csv')
df2 = pd.read_csv('../datasets/relation_20180912_spaces.csv')
df = pd.concat([df1, df2], ignore_index=True)
labels = ['dist_left','dist_right',
          'dist_top','dist_bottom',
          'dist_cx', 'dist_cy', 'angle',
          'dist_cx_left', 'dist_cx_right',
          'dist_cy_top', 'dist_cy_bottom',
          'dist_bottomright_middletop', 'dist_bottomright_middlebottom',
          'dist_topright_middletop', 'dist_topright_middlebottom']
#labels = ['dist_left','dist_right','dist_top','dist_bottom','dist_cx', 'dist_cy', 'angle']
target = df[df.warning==False][["relationship","path"]]
data = df[df.warning==False][labels]

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, shuffle=True, random_state=42)

train_path = y_train['path']
y_train = y_train['relationship']
test_path = y_test['path']
y_test = y_test['relationship']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%% Search best params
def grid_search_svc():
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

#%% 
svm = SVC(kernel='rbf', C=1.0, gamma=0.1)

svm.fit(X_train_scaled, y_train)

test_pred = svm.predict(X_test_scaled)
train_pred = svm.predict(X_train_scaled)

print('score train',svm.score(X_train_scaled, y_train))
print('score test',svm.score(X_test_scaled, y_test))
print(classification_report(y_train, train_pred, digits=3))
print(classification_report(y_test, test_pred, digits=3))


#%%

from scipy.spatial import Delaunay, Voronoi
import itertools

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            
            if far_point.tolist() not in new_vertices:
                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())
            else:
                new_region.append(new_vertices.index(far_point.tolist()))

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    #np.array(regions)[np.argsort(vor.point_region)].tolist()
    return new_regions, np.asarray(new_vertices)


def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def show(im):
    plt.imshow(im, origin='lower', cmap=cm.binary_r)
    plt.show()

def _finditem(obj, key):
    if key in obj: return obj[key]
    for k, v in obj.items():
        if isinstance(v,dict):
            item = _finditem(v, key)
            if item is not None:
                return item

def pairs(l):
    return zip(l[:-1], l[1:])

def formula_compounds(keys, level):
    im = np.sum([bboxes[k]['marker_full'] for k in keys], 0).astype(np.uint8)
    hist = cv2.reduce(im, (level+1)%2, cv2.REDUCE_MAX).reshape(-1).astype(np.float)
    blanks = np.array([i for i,n in enumerate(hist) if n==0])
    steps = [0]+[i for i,n in enumerate(abs(blanks[1:]-blanks[:-1])) if n>1]+[None]
    bounds = [np.mean(blanks[u:v]) for u,v in pairs(steps)]
    compounds = [[i for i,cx,cy,l in [coords[k] for k in keys] if (m < (l,cy)[level%2==0] < n)] for m,n in pairs(bounds)]
#    plt.figure()
#    plt.imshow(im, origin='lower', cmap=cm.binary)
#    for l in bounds:
#        if level%2==0:
#            plt.axhline(l)
#        else:
#            plt.axvline(l)
    return compounds

import networkx as nx
from collections import defaultdict


def read_struct(struct, nodes):
    return exec('struct{}'.format(''.join(['[{}]'.format(i) for i in nodes])))

def update_struct(struct, nodes):
    return exec('struct{}'.format(''.join(['[{}]'.format(i) for i in nodes])))

def nested_dict():
    return defaultdict(nested_dict)

def ppc(bboxes):
    keys = [k for k in bboxes.keys()]
    struct = nested_dict()
    
    leaves = []
    n_leaves = len(keys)
    level = 0
    comps_id = [0]
    struct['comps_list'] = [0]
    struct['levels'] = {0:{'comps':{'root':[0]}}}
    struct[0] = {'keys':keys}
    while len(leaves)<n_leaves:
        for parent in struct['levels'][level]['comps'].keys():
            for c in struct['levels'][level]['comps'][parent]:
                subcomps = formula_compounds(struct[c]['keys'], level)
                print(subcomps)
                if (struct[c]['keys'] in subcomps) and level>0:
                    i_ne = [len(x)!=0 for x in subcomps].index(True)
                    subcomps = [subcomps[i_ne][:1], subcomps[i_ne][1:]]
                n_comps = len(subcomps)
                cid = struct['comps_list'][-1]+1
                
                subcomps_id = list(range(cid, cid+n_comps))
                struct['comps_list']= struct['comps_list']+subcomps_id
                for scid, sc in zip(subcomps_id,subcomps):
                    try:
                        struct['levels'][level+1]
                    except:
                        struct['levels'][level+1] = {}
                    if len(sc)==1:
                        leaves.append((level,c,scid, sc[0]))
                        comp_type = 'elems'
                        #struct['levels'][level+1]['elems'] = struct['levels'][level+1].get('elems',{}).get('')+[scid]
                    else:
                        comp_type = 'comps'
                        #struct['levels'][level+1]['comps'] = struct['levels'][level+1].get('comps',[])+[scid]
                    
                    try:
                        struct['levels'][level+1][comp_type]
                    except:
                        struct['levels'][level+1][comp_type] = {}
                    
                    struct['levels'][level+1][comp_type][c] = struct['levels'][level+1][comp_type].get(c,[])+[scid]
                    struct[scid]={'keys':np.ravel(sc).tolist()}
        try:
            struct['levels'][level+1]['comps']
        except:
            break
        level += 1
    return struct, leaves


def create_graph(struct):
    F = nx.DiGraph()
    cid = 0
    for k,v in struct['levels'].items():
        for key in v.keys():
            for i,j in v[key].items():
                F.add_edges_from([(i,n) for n in j])

    mapping={} #{0:'a',1:'b',2:'c'}
    cid = 0
    for n in list(F.nodes())[1:]:
        if len(struct[n]['keys'])!=1:
            newname='comp_{}'.format(cid)
            cid+=1
        else:
            newname = np.ravel(struct[n]['keys'])[0]
        mapping[n] = newname
            
    G = nx.relabel_nodes(F,mapping)
    
    return G

def show_tree(graph):
    tree = nx.bfs_tree(graph, "comp_0")
    positions = nx.drawing.nx_agraph.graphviz_layout(tree, prog="dot")
    node_type = [True if degree[1]>1 else False for degree in tree.degree()]
    node_colors = ['red']+['blue' if k else 'green' for k in node_type][1:]
    node_sizes = [800]+[500 if k else 200 for k in node_type][1:]
    nx.draw_networkx_nodes(tree, positions, node_color=node_colors, alpha= 0.2, node_size=node_sizes)
    nx.draw_networkx_labels(tree, positions, font_size=8)
    nx.draw_networkx_edges(tree, positions, alpha=0.5)


def descendants(F, node):
    desc = [[n] if (not isinstance(n, str)) else [sn for sn in nx.nodes(nx.dfs_tree(F, n)) if not isinstance(sn, str)] for n in F.successors(node)]
    return np.concatenate(desc)

def successors(F, node):
    node_list = [n for n in F.successors(node)]
#    node_list = {'comp': [n for n in F.successors(node) if isinstance(n, str)],
#     'elem': [n for n in F.successors(node) if not isinstance(n, str)],
#     'all_elem': [n for n in nx.nodes(nx.dfs_tree(F, node)) if not isinstance(n, str)]}
    return node_list

def comps_level(F):
    comps_length = len([n for n in nx.nodes(nx.dfs_tree(F, 'comp_0')) if isinstance(n, str)])
    l = [['comp_0']]
    length = 0
    while length<comps_length:
        l.append([])
        for c in l[-2]:
            l[-1]+=[n for n in F.successors(c) if isinstance(n, str)]
            length += len(l[-1])
    return l

def ravel_comps(F):
    comps = {}
    comp_ids = [n for n in nx.nodes(nx.dfs_tree(F, 'comp_0')) if isinstance(n, str)]
    for c in comp_ids:
        comps[c] = [n for n in F.successors(c)]
    return comps

def globbox(bboxes, elems):
    x = min([bboxes[e]['left'] for e in elems])
    y = min([bboxes[e]['bottom'] for e in elems])
    w = max([bboxes[e]['right'] for e in elems])-x
    h = max([bboxes[e]['top'] for e in elems])-y
    bbox = {'marker_full':sum([bboxes[e]['marker_full'] for e in elems], 0),
                         'box':(x,y,w,h), 'centerx':x+0.5*w,
                         'centery':y+0.5*h, 'top':y+h,
                         'bottom':y, 'right':x+w,
                         'left':x, 'edges':np.array([[[x,y],[x,y+h]],
                                                     [[x,y],[x+w,y]],
                                                     [[x,y+h],[x+w,y+h]],
                                                     [[x+w,y],[x+w,y+h]]]),
                         'corners':np.array([[x,y],
                                             [x,y+h],
                                             [x+w,y],
                                             [x+w,y+h]])}
    return bbox

#%%
from time import time

t_start = time()
fpath =  "../datasets/Formulas/formulas-data/t4.png"

max_area = min(im.shape)//9
im, bboxes = test_formula(fpath, areas = range(1,27))

points = np.array([[bboxes[i]["centerx"], bboxes[i]["centery"]] for i in bboxes.keys()])
coords = [[i]+[bboxes[i][k] for k in ['centerx','centery','left']] for i in bboxes.keys()]

struct, leaves = ppc(bboxes)

print(time()-t_start)
F = create_graph(struct)
show_tree(F)

comp_bboxes = bboxes.copy()

for cname in np.concatenate(comps_level(F)).tolist()[::-1]:
        comp_bboxes[cname] = globbox(comp_bboxes, descendants(F, cname))

ordered_comps = {}
for cname in np.concatenate(comps_level(F)).tolist():
    subs = []
    succs = list(F.successors(cname))
    for subc in succs:
        subc_coords = [comp_bboxes[subc][k] for k in ['centerx','centery','left']]
        subs.append(subc_coords)
    ordered_comps[cname] = [succs[i] for i in [i[0] for i in sorted(enumerate(subs), key=lambda x: (x[1][2], x[1][1]))]]




fig, ax = plt.subplots()
ax.imshow(im, origin='lower', cmap=cm.binary_r)
ax.scatter(points[:,0], points[:,1])

for i, txt in enumerate(range(len(points))):
    ax.annotate(txt, (points[i,0]+5, points[i,1]+5), color='r')


for i in bboxes.keys():
    for n in direct_neighbors[i]:
        features = np.array(extract_features(bboxes[i], bboxes[n])).reshape(1,-1)
        features_scaled = scaler.transform(features)
        print(i,n)
        print(svm.predict(features_scaled))

#%%

# compute Voronoi tesselation
try:
    vor = Voronoi(points)
    tri = Delaunay(points)

    neighbors = {}
    for p in tri.vertices:
        for i,j in itertools.combinations(p,2):
            nj = ((j,), ())[j in neighbors.get(i,())]
            ni = ((i,), ())[i in neighbors.get(j,())]
            neighbors[i] = neighbors.get(i,()) + nj
            neighbors[j] = neighbors.get(j,()) + ni
            
    regions, vertices = voronoi_finite_polygons_2d(vor)

    direct_neighbors = {}
    for p in neighbors.keys():
        for n in neighbors[p]:
            edge = list(set(regions[p]).intersection(regions[n]))
            if len(edge)==2:
                A = vor.points[p]
                B = vor.points[n]
                C,D = vertices[edge]
                if intersect(A,B,C,D):
                    an = ((n,), ())[n in direct_neighbors.get(p,())]
                    ap = ((p,), ())[p in direct_neighbors.get(n,())]
                    direct_neighbors[p] = direct_neighbors.get(p,()) + an
                    direct_neighbors[n] = direct_neighbors.get(n,()) + ap

    fov_neighbors = {}
    for p in neighbors.keys():
        A = vor.points[p]
        for n in neighbors[p]:
            mask = np.isin(neighbors[p],[n], invert=True)
            is_visible = True
            B = vor.points[n]
            for q in np.array(neighbors[p])[mask]:
                for edge in bboxes[q]['edges']:
                    C,D = edge
                    if intersect(A,B,C,D):
                        is_visible = False
                        break
            if is_visible:
                an = ((n,), ())[n in fov_neighbors.get(p,())]
                ap = ((p,), ())[p in fov_neighbors.get(n,())]
                fov_neighbors[p] = fov_neighbors.get(p,()) + an
                fov_neighbors[n] = fov_neighbors.get(n,()) + ap
except:
    fov_neighbors = direct_neighbors = {0:(1,),1:(0,)}


for i in bboxes.keys():
    for j in direct_neighbors[i]:
        if j>i:
            ax.plot(points[[i,j],0], points[[i,j],1],'k')

for i in bboxes.keys():
    for n in direct_neighbors[i]:
        features = np.array(extract_features(bboxes[i], bboxes[n])).reshape(1,-1)
        features_scaled = scaler.transform(features)
        print(i,n)
        print(svm.predict(features_scaled))



# colorize
#for region in regions[:3]:
#    polygon = vertices[region]
#    plt.fill(*zip(*polygon), alpha=0.4)
#
#plt.plot(points[:,0], points[:,1], 'ko')
#plt.xlim(vor.min_bound[0] - 0.1, vor.max_bound[0] + 0.1)
#plt.ylim(vor.min_bound[1] - 0.1, vor.max_bound[1] + 0.1)
#
#plt.show()

#for region in np.array(regions).tolist():
#    polygon = vertices[region+region[:1]]
#    plt.plot(*zip(*polygon))