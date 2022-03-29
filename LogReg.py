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
from scipy import sparse 
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
            if int(row[0]) >= 6000:
                break
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

    unique_classes = len(np.unique([y[0] for y in Y_np.tolist()]))

    delta = np.zeros((unique_classes,rows))
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
    print("Done w/ conv\n")
    return X_sparse, Y_sparse, delta_sparse, unique_classes
    #return X, Y, delta_sparse
    
    #Prelim Grad descent. Need to work out P(Y|W,X) and lambda
    #need to get delta update implemented (I believe using sigma func.)
    # Having shape issues when doing dense-sparse mat mult with csr_mat. have NumPy set up though

#As written (the non-commented bits) assumes sparse matrix inputs
def grad_descent(X, Y, unique_classes, delta, lamb, learning_rate, iterations):
    print("Start GD")
    rows, columns = X.shape
    X_t = X.transpose()
    W_sparse = sparse.rand(unique_classes,columns)

    #W = np.random.rand(unique_classes,columns)
    
    Psparse = W_sparse.dot(X_t)
    Psparse = np.exp(Psparse)
    Psparse = Psparse.tolil()
    Psparse[-1, :] = 1
    Psparse = Psparse.tocsr()
    Psparse = normalize(Psparse,norm = 'l2')
    L_W = W_sparse.multiply(lamb)
    delta_P = ((delta-Psparse).dot(X))
    
    
    #This is the memory killer and where Grad desc is using all memory
    L_delta = delta_P.multiply(learning_rate)
    print("9")
    W_sparse = W_sparse + L_delta-L_W
    print("iter one done ")

    for i in range(1,iterations):
        print("iter", i, "Size of W=", W_sparse.shape)
        # Ps = np.matmul(W,(X.T))
        # Ps = np.exp(Ps)
        Psparse = W_sparse.dot(X_t)
        #Set bottom row to 1's and normalize each column
        #Psparse = Psparse.tolil()
        #Psparse[-1, :] = 1
        #Psparse = Psparse.tocsr()
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
    print("Done w/ GD\n")
    print(W_sparse.todense())
    return W_sparse


def classify(file, Y, W, K):
    """
    Classifies testing data using set of weights W. 
    Uses equations 27 and 28 from Mitchell Ch.3.

    Parameters
    ----------
    file : str
        Filename of the test data

    Y : list
        List of unique classes

    W : list
        List of weights
    """
    print("Y",Y)
    with open(file, newline='') as test_data:
        with open('solution.csv', 'w', newline='') as solution:
            data_reader = csv.reader(test_data)
            solout = csv.writer(solution)
            solout.writerow(['id', 'class'])

            n = W.shape[1] - 1
            W_1 = W[:,1:]
            
            count = 1
            for example in data_reader:
                print("Classifying row", count)
                count += 1

                X = example[1:]
                X = [int(x) for x in X]
                X = csr_matrix(X)

                #Column of k sums from 1->n of W[k][i] * X[i]
                sum_mat = W_1.dot(X.transpose())

                #List of P(Y=yk | X), max probability is the classifier
                probs = []
                for k in range(K+1):
                    n_sum = sum_mat[k, 0]

                    #Separate equations for k == K and k != K
                    if k != K:
                        num = np.exp(W[k, 0] + n_sum)
                        denom = 1
                        for j in range(K-1):
                            denom += np.exp(W[j, 0] + n_sum)
            
                        p = num / denom
                        probs.append(p)
                    else:
                        denom = 1
                        for j in range(K-1):
                            denom += np.exp(W[j, 0] + n_sum)

                        p = 1 / denom
                        probs.append(p)

                idx = np.argmax(probs)
                ans = [int(example[0]), idx]

                #Print answer pairs and write to csv
                #print(ans)
                solout.writerow(ans)

if __name__ == "__main__":
    # results = to_sparse_mat("/home/jared/Downloads/training.csv")
    results = to_sparse_mat("training.csv")

    W = grad_descent(results[0], results[1], results[3], results[2], .001, .001, 1000)
    print("W:", W.shape)

    #Converting Y formatting for classification
    Y = results[1].todense().tolist()[0]
    Y = [[y] for y in Y]
    K = results[3] - 1
    classify("testing.csv", Y, W, K)
