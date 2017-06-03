# -*- coding: utf-8 -*-
"""
Created on Mon May 22 07:58:56 2017

@author: ykhoja
"""

import numpy as np
from numpy import linalg as LA
import matplotlib as plt
import networkx as nx


def delete_row(W,x):
    #issue with the case when W and x are identical, and its supposed to return an empty array [[]]
    if(np.all(W != x)):
        return W
    #elif(np.all(W == x) and W.size == d):
    #    return np.array([[]])
    else:
        idx_delete = np.where(W == x)[0][0] #get index of row to delete
        return np.delete(W,idx_delete,axis=0) #delete row

#Create Near(V,x,r) function which finds N = {v in V | ||x- v||<r}
#If the N is empty, it returns a vector of -1
def Near(V,x,r):
    tmp = delete_row(V,x)
    N = np.empty(d) #initialize an empty array
    pad = N
    for row in tmp:   
        if(LA.norm(row - x) < r and np.all(row != x)):
            N = np.vstack((N,row))
    N = delete_row(N,pad)
    return N

def intersect(A, B):
    pad = np.ones(d)*-1 #add a padding row to ensure all arrays have more than two rows
    A = np.vstack((A,pad))
    B = np.vstack((B,pad))
    aset = set([tuple(x) for x in A])    
    bset = set([tuple(x) for x in B])
    C = np.array([x for x in aset & bset])
    C = delete_row(C,pad) #remove padding row
    return C

def argmin_cost(Y_near,x,mode):
    dist = []
    if(mode == 1):
        for y in Y_near:
            dist.append(LA.norm(x_init-y) + LA.norm(y-x))
        i = np.argmin(dist)
    if(mode == 2):
        for y in Y_near:
            dist.append(LA.norm(x_init-y))
        i = np.argmin(dist)
    return Y_near[i]

#Initialize random sample
d = 2 # dimension of the Euclidean space
x_init =  np.ones(d)*0.5; # set to center of configuration space
k = 300; #sample size of X_free
sample = np.random.uniform(0,1,size=(k,d));
V = np.vstack((sample,x_init)); # {x_init} U sample with x_init as the last row
E = np.ones(d)*-1; #
H = x_init; # Create set of used vertices
z = x_init;
r_n = 0.1 # Parameter of the Near( ) function
X_goal = np.ones(d)*.99
W = delete_row(V,x_init) # Create set of unused vertices by removing x_init
N_z = Near(V,z,r_n)
cnt = 0
while z not in X_goal:
    print("while loop running")
    H_new = np.ones(d)*-1
    X_near = intersect(N_z,W)
    if(cnt == 3):
        break
    for x in X_near:
        print("started for loop")
        N_x = Near(V,x,r_n)
        Y_near = intersect(N_x,H)
        y_min = argmin_cost(Y_near,x, 1) #set mode to 1 to find total cost
        #SKIP CollisionFree BUT INSERT HERE LATER
        E = np.vstack((E,[y_min,x]))
        E = delete_row(E,np.ones(d)*-1)
        H_new = np.vstack((H_new,x))
        H_new = delete_row(H_new,np.ones(d)*-1)
        W = delete_row(W,x)
        #IF statement of CollisionFree ends here
        print("ended for loop")
    print("H_new = ",H_new)
    H_new = delete_row(H_new,np.ones(d)*-1)
    print("H_new = ",H_new)
    H = np.vstack((H,H_new))
    H = delete_row(H,z)
    z = argmin_cost(H,[], 2) #set mode to 2 to find cost based on tree only
    cnt = cnt + 1
    
print("while loop stopped")

#add part of the function that outputs the path in the tree from x_init to z
G = nx.path_graph(8)
