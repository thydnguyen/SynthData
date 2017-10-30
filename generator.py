# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:14:20 2017

@author: thy1995
"""

import numpy as np
from sklearn.metrics.pairwise import  pairwise_distances
x


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
    cluster_std = np.ones(len(centers)) * cluster_std
    if not even_std :
        cluster_std = cluster_std + np.clip(np.random.normal(scale = cluster_std[0]  ,size = len(cluster_std)),0.01, None)        

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
    print(n_samples_per_center)
    for i, (n, std) in enumerate(zip(n_samples_per_center, cluster_std)):
        X.append(centers[i])
        X.extend((centers[i] + np.random.normal(scale=std,
                                               size=(n - 1, n_features))))
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
    
    max_cluster_std = cluster_std * (n_features ** 0.5) * 8
    min_cluster_std = cluster_std * (n_features ** 0.5) * 4
    lnspace = np.linspace(center_box[0], center_box[1], num =  100 )

    done =  False
    while not done :
        
        Centers = np.random.choice(lnspace, size = (even_centers, n_features))

        #print(distance)
        distance = pairwise_distances(Centers)
        upper = np.triu_indices_from(distance, 1)
        distance  = distance[upper]
        
        if min(distance) > max_cluster_std:
            done= True

        if uneven_centers > 0:
            target = np.random.choice(range(len(Centers)), size = (uneven_centers), replace = False)
            target = Centers[target] + np.random.normal(scale= cluster_std * 3 , size=(uneven_centers, n_features))
            
            Centers = np.append(Centers, target, axis = 0)


#        ratio = 1 - (np.sum(distance > max_cluster_std) / len(distance))
#        
#        print(ratio)
#        if ratio > fair_ratio:
#            done = True
        
        #print("even", even)
        #print("uneven", uneven)

            
    centers = Centers
    cluster_std = np.ones(len(centers)) * cluster_std
    
    if not even_std :
        cluster_std = cluster_std + np.clip(np.random.normal(scale = cluster_std[0] / 2 ,size = len(cluster_std)),0.01, None)    
    
    
    X = []
    y = []
    n_centers = centers.shape[0]
    n_samples_per_center = [n_samples] * n_centers
    if not even_mem:
        n_samples_per_center[0] = n_samples_per_center[0] * 4
#        for i in range(n_samples % n_centers):
#            n_samples_per_center[i] += 1
#    else:
#        skewed = int(n_samples // (n_centers+ 1)) * 2
#        left_over = n_samples - skewed
#        n_samples_per_center = [int(left_over // (n_centers - 1 ))] * (n_centers - 1)
#        for i in range(left_over % (n_centers - 1)):
#            n_samples_per_center[i] += 1
#        n_samples_per_center.append(skewed)
    print(n_samples_per_center)

    for i, (n, std) in enumerate(zip(n_samples_per_center, cluster_std)):
        X.append(centers[i])
        X.extend((centers[i] + np.random.normal(scale=std,
                                               size=(n - 1, n_features))))
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

    done =  False
    while not done :
        Centers = np.random.choice(lnspace, size = (centers, n_features))
        distance = pairwise_distances(Centers)
        upper = np.triu_indices_from(distance, 1)
        distance  = distance[upper]
        #print(distance)
        if len(np.unique(distance)) == 0 :
            continue
        
        
        even = sum(distance > max_cluster_std)
        uneven = sum(distance <= min_cluster_std)
#        print("even", even)
#        print("uneven", uneven)
        if even == even_num and uneven == uneven_num:
            done = True
            
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