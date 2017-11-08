# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 14:06:53 2017

@author: thy1995
"""
size = [50, 100, 200]
#size = [10]
attributes = [2,4,8,16,32]
#attributes = [2]
classes = range(2,11)

signature = 'eff'
equal = [True, False]
even_std = [True,False]
even_mem = [True, False]
noise = [True, False]

savefolder_d = "D:\\CLS_lab\\codeTest\\batchSynthetic\\dataNewSep\\"
savefolder_l = "D:\\CLS_lab\\codeTest\\batchSynthetic\\labelNewSep\\"
savefolder_g = "D:\\CLS_lab\\codeTest\\batchSynthetic\\graphNewSep\\"

from generatorHier import make_blobs, uneven_blobs
#from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import itertools
from fileOP import writeRows
import os

def makeFolder(addr):
    if not os.path.exists(addr):
        os.makedirs(addr)

makeFolder(savefolder_d)
makeFolder(savefolder_g)
makeFolder(savefolder_l)



pca = PCA(n_components= 2)


for s,a,c, e ,std ,m,n in itertools.product(size, attributes, classes, equal, even_std, even_mem, noise):
    print("size is: ",s)
    print("attribute is: ", a)
    print("classes is: ",c )
    
    if std == False and m == False:
        continue
    
    signature = ''
    if e:
        signature = signature + 'e'
    else:
        signature = signature + 'u'
        
    if std:
        signature = signature + 't'
    else:
        signature = signature + 'f'
    
    if m:
        signature = signature + 't'
    else:
        signature = signature + 'f'


    if e:
        data , label = make_blobs(n_samples = s, n_features= a, centers = c, center_box = [-10,10], cluster_std= 0.5, even_std = std, even_mem = m, noise = n)
    else:
        data , label = uneven_blobs(n_samples = s, n_features= a, centers = c, center_box = [-10,10], cluster_std= 0.5, even_std = std, even_mem = m,noise = n)
    
    label = label + 1
    writeRows(savefolder_d + 's' + str(s) + 'a' + str(a) + 'c' + str(c) + signature + '.csv', data)
    writeRows(savefolder_l + 's' + str(s) + 'a' + str(a) + 'c' + str(c) + signature +'.csv', np.transpose(np.expand_dims(label, axis = 0)).tolist())
    X1, X2  = np.transpose(pca.fit_transform(data))
    unique = np.unique(label)
    for i in unique:
        X_r1 = [X1[j] for j in range(len(X1)) if label[j] == i]
        X_r2 = [X2[j] for j in range(len(X2)) if label[j] == i]
        l = [label[j] for j in range(len(label)) if label[j] == i]
        plt.scatter(X_r1, X_r2, label = label)
    plt.savefig(savefolder_g + 's' + str(s) + 'a' + str(a) + 'c' + str(c) + signature)
    plt.clf()

