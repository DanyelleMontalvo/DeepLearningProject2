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
import pandas as pd
from sklearn.preprocessing import normalize


def to_sparse_mat(file):
    
    col_count = []
    row_count = []
    data_count = []
    rowcount = 0
    #code incorporated from from https://docs.python.org/3/library/csv.html
    #Thanks Danyelle (:
    with open(file, newline='', encoding='utf-8-sig') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            for idx, el in enumerate(row):
                if int(el) == 0:
                    continue
                else:
                    row_count.append(rowcount)
                    col_count.append(idx)
                    data_count.append(int(el))
            rowcount = rowcount + 1
    new_matrix = csr_matrix((data_count, (row_count, col_count)), shape=(rowcount, 61190))
    numpy_array = new_matrix.todense()

    #Build Y as described in Proj. 2 description
    Y_np = numpy_array[:,-1]
    numpy_array2 = numpy_array[:,:-1]
    rows, columns = numpy_array.shape
    unique_classes = (np.unique(Y_np)).shape
    unique_classes = unique_classes[1]
    delta=np.zeros((unique_classes,rows))
    for r in numpy_array[:,-1]:
        for c in range(0,rows):
            if(r==numpy_array[c,-1]):
                delta[r-1,c] =1
    #numpy_array2 = numpy_array[:,:-1]
    #rows, columns = numpy_array.shape
    ones = np.ones((rows,1))
    

    #X as described in Proj. 2 description
    numpy_array2 = numpy_array2[:,1:]

    final_array = np.append(ones,numpy_array2,1)
    #Normalized array
    #k as described in Proj. 2 description
    #init delta as described in Proj. 2 description (k x m matrix of zeros)
    
    #Sparse versions of matrices
    X_sparse = csr_matrix(final_array)
    Y_sparse = csr_matrix(Y_np.T)
    delta_sparse = csr_matrix(delta)
    
    #NumPy versions of Matrices. These handled dense-sparse multiplication better. Will look into further
    # X = (final_array)
    # Y = (Y_np.T)
    # delta_sparse = (delta)
   # print(X_sparse)
    return X_sparse, Y_sparse, delta_sparse, unique_classes
    #return X, Y, delta_sparse
    
    #Prelim Grad descent. Need to work out P(Y|W,X) and lambda
    #need to get delta update implemented (I believe using sigma func.)
    # Having shape issues when doing dense-sparse mat mult with csr_mat. have NumPy set up though

#As written (the non-commented bits) assumes sparse matrix inputs
def grad_descent(X, Y, unique_classes, delta, lamb, learning_rate, iterations):
    rows, columns = X.shape
    
    W = np.random.rand(unique_classes,columns)
    W_sparse = csr_matrix(W)
    for i in range(0,iterations):
        # Ps = np.matmul(W,(X.T))
        # Ps = np.exp(Ps)
        Psparse = W_sparse.dot(X.T)
        #Set bottom row to 1's and normalize each column
        Psparse = Psparse.tolil()
        Psparse[-1, :] = 1
        Psparse = Psparse.tocsr()
       # Ps[len(Ps)-1] = np.ones_like(Ps[0])
        #Sparse normalize
        Psparse = normalize(Psparse,norm = 'l2')
        # for col in range(len(Ps[0])):
        #     sum = 0
        #     for i in range(len(Ps)):
        #         sum += Ps[i][col]
        #     for i in range(len(Ps)):
        #         Ps[i][col] /= sum

        #W = W + learning_rate*(np.matmul(delta-Ps,X)-lamb*W)
        W_sparse = W_sparse + learning_rate*(((delta-Psparse).dot(X))-lamb*W_sparse)

       

    return W_sparse



def classify(file, Y, W):
    #I'm pretty sure this is necessary based on below implementation
    Y = Y.todense()
    W = W.todense()
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
    results = to_sparse_mat("/home/jared/Downloads/training.csv")
    W = grad_descent(results[0], results[1], results[3], results[2], .001, .001, 10)
    classify("/home/jared/Downloads/training.csv", results[1], W)