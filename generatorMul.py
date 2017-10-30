# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 15:02:53 2017

@author: thy1995
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:14:20 2017

@author: thy1995
"""

import numpy as np
from scipy import random
from sklearn.metrics.pairwise import  pairwise_distances
from sklearn.datasets import make_spd_matrix
from sklearn.utils.extmath import cartesian
from itertools import combinations
from random import shuffle as Shuffle
from scipy.spatial import ConvexHull



def make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=0.5,
               center_box=(-10.0, 10.0), shuffle=True, random_state=None, even_std = True, even_mem= True):
    


    max_cluster_std = cluster_std * (n_features ** 0.5) * 8
    lnspace = np.linspace(center_box[0], center_box[1], num =  10)
    smallest_distance = 0 
    while smallest_distance < max_cluster_std:
        Centers = np.random.choice(lnspace, size = (centers, n_features))
        distance = pairwise_distances(Centers)
        upper = np.triu_indices_from(distance, 1)
        distance  = np.sort(distance[upper])[0]

        if distance > smallest_distance:
            smallest_distance = distance 
            print(smallest_distance)
    centers = Centers
    
   

    X = []
    y = []
    n_centers = centers.shape[0]
 
        
    n_samples_per_center = [n_samples] * n_centers
    if not even_mem:
        n_samples_per_center[0] = n_samples_per_center[0]
#        for i in range(n_samples % n_centers):
#            n_samples_per_center[i] += 1
#    else:
#        skewed = int(n_samples // (n_centers+ 1)) * 2
#        left_over = n_samples - skewed
#        n_samples_per_center = [int(left_over // (n_centers - 1 ))] * (n_centers - 1)
#        for i in range(left_over % (n_centers - 1)):
#            n_samples_per_center[i] += 1
#        n_samples_per_center.append(skewed)
    cluster_std = []
    cov = make_spd_matrix(n_features)
    if even_std:
        cluster_std  = [cov] * n_centers
    else:
        for _ in range(n_centers):
            cluster_std.append(make_spd_matrix(n_features))

    print(n_samples_per_center)

    for i, (n, std) in enumerate(zip(n_samples_per_center, cluster_std)):
        print(std)
        X.append(centers[i])

   
        X.extend((centers[i] + np.random.multivariate_normal(mean = np.zeros(n_features), cov = std,
                                               size=(n - 1,))))
        y += [i] * n
        
    #X = np.concatenate(X)
    y = np.array(y)


    return X, y

def uneven_blobs(n_samples=100, n_features=2, centers=3, cluster_std=0.5,
               center_box=(-10.0, 10.0), shuffle=True, random_state=None, even_std = True, even_mem = True):
    
    fair_ratio = 1/3
    
    dist_count = sum(range(centers))
    even_centers = int(np.max([centers - 2 , 2]))
    uneven_centers = centers - even_centers
    even_num  = sum(range(even_centers))
    uneven_num = dist_count - even_num
    
    max_cluster_std = cluster_std * (n_features ** 0.5) * 4
    min_cluster_std = cluster_std * (n_features ** 0.5) * 2
    #lnspace = np.linspace(center_box[0], center_box[1], num =  1000 )
#    
    #cart_matrix = cartesian([lnspace] * n_features]
    done = False
    count = 0
    
    
#    while not done:
#        print("Iteration", count)
#        count = count + 1
#        x = np.random.uniform(center_box[0], center_box[1],size=(30,n_features )) 
#        
#    #    x /= np.linalg.norm(x, axis=1)[:, np.newaxis]
#    #    x = x * 10 
#    #    all_combinations = list(combinations(x, centers))
#        
#        all_combinations = list(combinations(x, centers))
#        Shuffle(all_combinations)
#        print("length is", len(all_combinations))
#        
#        Centers = []
#        for candidate in all_combinations:
#            distance = pairwise_distances(candidate)
#            upper = np.triu_indices_from(distance, 1)
#            distance  = distance[upper]
#            if all(distance > min_cluster_std) and all(distance < max_cluster_std):
#                Centers = candidate
#                break
#        if Centers == []:
#            print("No satisfying centers constraint")
#            done = False
#        else:
#            done = True
    Centers = np.array(Centers)
    centers = Centers
    print(centers)
    cluster_std = np.ones(len(centers)) * cluster_std
            
    X = []
    y = []
    n_centers = centers.shape[0]
    n_samples_per_center = [n_samples] * n_centers
    
    if not even_mem:
        n_samples_per_center[0] = n_samples_per_center[0]
        
    cov = make_spd_matrix(n_features)
    cluster_std = []
    if even_std:
        cluster_std  = [cov] * n_centers
    else:
        for _ in range(n_centers):
            cluster_std.append(make_spd_matrix(n_features))



    for i, (n, std) in enumerate(zip(n_samples_per_center, cluster_std)):

        X.append(centers[i])

   
        X.extend((centers[i] + np.random.multivariate_normal(mean = np.zeros(n_features), cov = std,
                                               size=(n - 1, ))))
        y += [i] * n
        
    #X = np.concatenate(X)
    y = np.array(y)


    return X, y




def uneven_blobs_dep(n_samples=100, n_features=2, centers=3, cluster_std=0.5,
               center_box=(-10.0, 10.0), shuffle=True, random_state=None):
    #DEPRECATION VERSION
    dist_count = sum(range(centers))
    uneven_num = np.round(dist_count / 4) if np.round(dist_count / 4) > 0 else 1
    even_num  = dist_count - uneven_num
    
    
    print("required even:", even_num)
    print("required uneven:", uneven_num)

    max_cluster_std = cluster_std * (n_features ** 0.5) * 8
    min_cluster_std = cluster_std * (n_features ** 0.5) * 6
    lnspace = np.linspace(center_box[0], center_box[1], num =  100 )

    cart_matrix = cartesian([lnspace] * n_features)
    all_combinations = combinations(cart_matrix, centers)
    Centers = []
    for candidate in all_combinations:
        distance = pairwise_distances(candidate)
        upper = np.triu_indices_from(distance, 1)
        distance  = distance[upper]
        if all(distance > min_cluster_std) and all(distance < max_cluster_std):
            Centers = candidate
            break
        
    if Centers == []:
        print("No satisfying centers constraint")
        return 0
            
    centers = Centers
    cluster_std = np.ones(len(centers)) * cluster_std

    X = []
    y = []
    n_centers = centers.shape[0]
    n_samples_per_center = [int(n_samples // n_centers)] * n_centers

    for i in range(n_samples % n_centers):
        n_samples_per_center[i] += 1

    for i, (n, std) in enumerate(zip(n_samples_per_center, cluster_std)):
        X.append(centers[i] + np.random.normal(scale=std,
                                               size=(n, n_features)))
        y += [i] * n

    X = np.concatenate(X)
    y = np.array(y)


    return X, y

def findCentroid(data, label):
    label_unique = np.unique(label)
    centroid_list = np.array([])
    for i in label_unique:
        temp = [data[j] for j in range(len(label)) if label[j] == i]
        temp = np.average(temp, axis = 0)

def NoisyDataset(n_sample,n_attribute, n_classes, n_noisy):
    sample_per_center = int(n_sample / n_classes)
    data , label = make_blobs(n_samples = sample_per_center, n_features= n_attribute, centers = n_classes) 