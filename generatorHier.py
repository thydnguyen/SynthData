# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 14:21:02 2017

@author: thy1995
"""

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
import scipy
import random as Random
from sklearn.metrics.pairwise import  pairwise_distances
from sklearn.datasets import make_spd_matrix
from sklearn.utils.extmath import cartesian
from itertools import combinations
from random import shuffle as Shuffle


def make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=0.5,
               center_box=(-10.0, 10.0), shuffle=True, random_state=None, even_std = True, even_mem = True):
    separate_factor = 6
    
    cov = make_spd_matrix(n_features)
    Cluster_std = []
    if even_std:
        Cluster_std  = [cov] * centers
    else:
        for _ in range(centers):
                Cluster_std.append(make_spd_matrix(n_features))
    
    fair_ratio = 1/3
    
    dist_count = sum(range(centers))
    even_centers = int(np.max([centers - 2 , 2]))
    uneven_centers = centers - even_centers
    even_num  = sum(range(even_centers))
    uneven_num = dist_count - even_num
    
    #max_cluster_std =  (n_features ** 0.5) * 6
    #min_cluster_std =  (n_features ** 0.5) 
    
    Centers = [np.random.uniform(center_box[0], center_box[1], size = (1,n_features))[0]]
    for i in range(centers - 1):
        done = False
        while not done:
            candidateSet = Centers[:]
            randomIndex = Random.choice(range(len(candidateSet)))
            targetCluster = candidateSet[randomIndex]
            targetCov  = Cluster_std[randomIndex]
            target_std  = np.std(np.random.multivariate_normal(targetCluster, targetCov, size = n_samples), axis = 0)
            target_var  = np.var(np.random.multivariate_normal(targetCluster, targetCov, size = n_samples), axis = 0)
            
            
            min_cluster_std = np.sqrt(np.sum(target_var)) * separate_factor
            max_cluster_std = np.sqrt(np.sum(target_var)) * (separate_factor + 0.5)
            
            newCandidateCluster = targetCluster + (target_std * np.random.normal(scale = 0.5, size = n_features ) + target_std) * separate_factor 
            candidateSet.append(newCandidateCluster)
            minDistance =  scipy.spatial.distance.cdist([targetCluster], [newCandidateCluster])[0][0]
            newMinDistance  = np.min(scipy.spatial.distance.cdist(Centers, [newCandidateCluster]))
            print("Required min:", min_cluster_std)
            print("Required max:", max_cluster_std)
            print("Current min", newMinDistance)
#            distance = pairwise_distances(candidateSet)
#            upper = np.triu_indices_from(distance, 1)
#            distance  = distance[upper]
#            #print(t)
#            print("Max distance:",max(distance))
#            print("Required distance", max_cluster_std)
            #average_distance = [np.mean(np.random.multivariate_normal(np.zeros(n_features), targetCov, size = n_samples))
            if   minDistance > min_cluster_std and minDistance < max_cluster_std and minDistance == newMinDistance:
                print("Finished")
                done = True
                Centers = candidateSet
            
    Centers = np.array(Centers)
    centers = Centers


            
    X = []
    y = []
    n_centers = centers.shape[0]
    n_samples_per_center = [n_samples] * n_centers
    
    if not even_mem:
        n_samples_per_center = [n_samples_per_center[i] * (i+1) for  i in range(len(n_samples_per_center)) ]
        




    for i, (n, std) in enumerate(zip(n_samples_per_center, Cluster_std)):

        X.append(centers[i])

   
        X.extend((centers[i] + np.random.multivariate_normal(mean = np.zeros(n_features), cov = std,
                                               size=(n - 1, ))))
        y += [i] * n
        
    #X = np.concatenate(X)
    y = np.array(y)


    return X, y


def uneven_blobs(n_samples=100, n_features=2, centers=3, cluster_std=0.5,
               center_box=(-10.0, 10.0), shuffle=True, random_state=None, even_std = True, even_mem = True):
    separate_factor = 2
    cov = make_spd_matrix(n_features)
    Cluster_std = []
    if even_std:
        Cluster_std  = [cov] * centers
    else:
        for _ in range(centers):
            Cluster_std.append(make_spd_matrix(n_features))
    
    fair_ratio = 1/3
    
    dist_count = sum(range(centers))
    even_centers = int(np.max([centers - 2 , 2]))
    uneven_centers = centers - even_centers
    even_num  = sum(range(even_centers))
    uneven_num = dist_count - even_num
    
    #max_cluster_std =  (n_features ** 0.5) * 6
    #min_cluster_std =  (n_features ** 0.5) 
    
    Centers = [np.random.uniform(center_box[0], center_box[1], size = (1,n_features))[0]]
    for i in range(centers - 1):
        done = False
        while not done:
            candidateSet = Centers[:]
            randomIndex = Random.choice(range(len(candidateSet)))
            targetCluster = candidateSet[randomIndex]
            targetCov  = Cluster_std[randomIndex]
            target_std  = np.std(np.random.multivariate_normal(targetCluster, targetCov, size = n_samples), axis = 0)
            target_var  = np.var(np.random.multivariate_normal(targetCluster, targetCov, size = n_samples), axis = 0)
            
            
            min_cluster_std = np.sqrt(np.sum(target_var)) * separate_factor
            max_cluster_std = np.sqrt(np.sum(target_var)) * (separate_factor + 0.5)
            
            newCandidateCluster = targetCluster + (target_std * np.random.normal(scale = 0.5, size = n_features ) + target_std) * separate_factor 
            candidateSet.append(newCandidateCluster)
            minDistance =  scipy.spatial.distance.cdist([targetCluster], [newCandidateCluster])[0][0]
            newMinDistance  = np.min(scipy.spatial.distance.cdist(Centers, [newCandidateCluster]))
            print("Required min:", min_cluster_std)
            print("Required max:", max_cluster_std)
            print("Current min", newMinDistance)
#            distance = pairwise_distances(candidateSet)
#            upper = np.triu_indices_from(distance, 1)
#            distance  = distance[upper]
#            #print(t)
#            print("Max distance:",max(distance))
#            print("Required distance", max_cluster_std)
            #average_distance = [np.mean(np.random.multivariate_normal(np.zeros(n_features), targetCov, size = n_samples))
            if   minDistance > min_cluster_std and minDistance < max_cluster_std and minDistance == newMinDistance:
                print("Finished")
                done = True
                Centers = candidateSet
            
    Centers = np.array(Centers)
    centers = Centers


            
    X = []
    y = []
    n_centers = centers.shape[0]
    n_samples_per_center = [n_samples] * n_centers
    
    if not even_mem:
        n_samples_per_center = [n_samples_per_center[i] * (i+1) for  i in range(len(n_samples_per_center)) ]
        




    for i, (n, std) in enumerate(zip(n_samples_per_center, Cluster_std)):

        X.append(centers[i])

   
        X.extend((centers[i] + np.random.multivariate_normal(mean = np.zeros(n_features), cov = std,
                                               size=(n - 1, ))))
        y += [i] * n
        
    #X = np.concatenate(X)
    y = np.array(y)


    return X, y


def make_blobs_dep(n_samples=100, n_features=2, centers=3, cluster_std=0.5,
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

