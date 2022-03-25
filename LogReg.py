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

    # with open(file, newline='') as csvfile:
    #     reader = csv.reader(csvfile)
    #     for row in reader:

    numpy_array = csv[:,1:]
    print(numpy_array)
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
    delta=np.ones((unique_classes,rows))
    
    #Sparse versions of matrices
    # X_sparse = csr_matrix(final_array)
    # Y_sparse = csr_matrix(Y_np.T)
    # delta_sparse = csr_matrix(delta)
    
    #NumPy versions of Matrices. These handled dense-sparse multiplication better. Will look into further
    X = (final_array)
    Y = (Y_np.T)
    delta_sparse = (delta)
    #return X_sparse, Y_sparse, delta_sparse
    return X, Y, delta_sparse
    
    #Prelim Grad descent. Need to work out P(Y|W,X) and lambda
    #need to get delta update implemented (I believe using sigma func.)
    # Having shape issues when doing dense-sparse mat mult with csr_mat. have NumPy set up though

def grad_descent(X, Y, delta, lamb, learning_rate, iterations):
    rows, columns = X.shape
    W = np.random.rand(len(np.unique(Y)),columns)
    #W_sparse = csr_matrix(W)
    for i in range(0,iterations):
        Ps = np.matmul(W,(X.T))
        Ps = np.exp(Ps)

        #Set bottom row to 1's and normalize each column
        Ps[len(Ps)-1] = np.ones_like(Ps[0])
        for col in range(len(Ps[0])):
            sum = 0
            for i in range(len(Ps)):
                sum += Ps[i][col]
            for i in range(len(Ps)):
                Ps[i][col] /= sum

        #Psparse = W_sparse.multiply(X.T)
        W = W + learning_rate*(np.matmul(delta-Ps,X)-lamb*W)
        #W_sparse = W_sparse + learning_rate*(((delta-Psparse).multiply(X))-lamb*W_sparse)

        #print("W ",i) 
        #print(W)

    return W


def classify(file, Y, W):
    Y = list(set(Y))
    
    with open(file, newline='') as test_data:
        with open('solution.csv', 'w', newline='') as solution:
            data_reader = csv.reader(test_data)
            solout = csv.writer(solution)
            solout.writerow(['id', 'class'])

            for example in data_reader:
                X = example[1:]
                X = [int(x) for x in X]

                #List of P(Y=yk | X), max probability is the classifier
                probs = []
                K = len(Y) - 1
                n = len(X) - 1
                for k, y in enumerate(Y):
                    if k != K:
                        num = np.exp(W[k][0] + sum(W[k][i] * X[i] for i in range(n)))
                        denom = 1
                        for j in range(K-1):
                            denom += np.exp(W[j][0] + sum(W[j][i] * X[i] for i in range(n)))
            
                        p = num / denom
                        probs.append(p)
                    else:
                        denom = 1
                        for j in range(K-1):
                            denom += np.exp(W[j][0] + sum(W[j][i] * X[i] for i in range(n)))

                        p = 1 / denom
                        probs.append(p)

                idx = np.argmax(probs)
                ans = [int(example[0]), int(Y[idx])]

                print(ans)
                solout.writerow(ans)

if __name__ == "__main__":
    results = to_sparse_mat("test_train.csv")
    W = grad_descent(results[0], results[1], results[2], .001, .001, 1000)
    classify("test_tester.csv", results[1], W)