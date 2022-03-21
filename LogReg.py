#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 23:08:51 2022

"""
from numpy import genfromtxt
import numpy as np
import csv
import statistics as st
from scipy.sparse import csr_matrix


def to_sparse_mat(file):
    csv = np.genfromtxt (file, delimiter=",")
    numpy_array = csv[:,1:]
    #Build Y as described in Proj. 2 description
    Y_np = numpy_array[:,-1]
    numpy_array = numpy_array[:,:-1]
    rows, columns = numpy_array.shape
    ones = np.ones((rows,1))
    #X as described in Proj. 2 description
    final_array = np.append(ones,numpy_array,1)
    #Normalized array
    final_array = final_array/final_array.sum(axis=0,keepdims=1)
    #k as described in Proj. 2 description
    unique_classes = len(np.unique(Y_np))
    #init delta as described in Proj. 2 description (k x m matrix of zeros)
    delta=np.zeros((unique_classes,rows))
    #Sparse versions of matrices
    # X = csr_matrix(final_array)
    # Y = csr_matrix(Y_np.T)
    # delta_sparse = csr_matrix(delta)
    
    #NumPy versions of Matrices. These handled dense-sparse multiplication better. Will look into further
    X = (final_array)
    Y = (Y_np.T)
    delta_sparse = (delta)
    return X, Y, delta_sparse
    
    #Prelim Grad descent. Need to work out P(Y|W,X) and lambda
def grad_descent(X, Y, delta, lamb, learning_rate, iterations):
    rows, columns = X.shape
    W = np.random.rand(len(np.unique(Y)),columns)
    for i in range(0,iterations):
        Ps = np.matmul(W,(X.T))
        W = W+ (learning_rate)*(np.matmul(delta-Ps,X)-lamb*W)
        print("W ",i) 
        print(W)
if __name__ == "__main__":
     results = to_sparse_mat("test.csv")
     grad_descent(results[0], results[1], results[2], .1, .001, 10)
