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
from scipy.sparse import lil_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy import sparse
from scipy import linalg 
import pandas as pd
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
import datetime
from collections import defaultdict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pprint import pprint
import matplotlib.pyplot as plt

def to_sparse_mat_for_conf(file,skipstart,stopafter):
    
    col_count = []
    row_count = []
    data_count = []
    rowcount = 0
    #code incorporated from from https://docs.python.org/3/library/csv.html
    #Thanks Danyelle (:
    with open(file, newline='', encoding='utf-8-sig') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        filecount = 0
        for row in csvreader:
            filecount = filecount+1
            if filecount < skipstart:
                continue
            if rowcount >= stopafter and stopafter > 0:
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
    #new_matrix = csr_matrix((data_count, (row_count, col_count)), shape=(rowcount, 5))
    numpy_array = new_matrix.todense()

    #Build Y as described in Proj. 2 description
    Y_np = numpy_array[:,-1]
    numpy_array = numpy_array[:,:-1]
    orig_X = numpy_array
    rows, columns = numpy_array.shape

    unique_classes = len(np.unique([y[0] for y in Y_np.tolist()]))

    delta = np.zeros((unique_classes,rows))
    #for r in numpy_array[:,-1]:
    #    for c in range(0,rows):
    #        if(r==numpy_array[c,-1]):
    #            delta[r-1,c] =1
    for c in range(0, rows):
        delta[Y_np[c]-1,c] = 1

    #numpy_array2 = numpy_array[:,:-1]
    #rows, columns = numpy_array.shape
    ones = np.ones((rows,1))
    
    #X as described in Proj. 2 description
    numpy_array = numpy_array[:,1:]

    final_array = np.append(ones,numpy_array,1)
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
    return X_sparse, Y_sparse, delta_sparse, unique_classes, Y_np, orig_X

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
            if int(row[0]) >= 12000:
            #if int(row[0]) >= 10:
  
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
    #new_matrix = csr_matrix((data_count, (row_count, col_count)), shape=(rowcount, 5))
    numpy_array = new_matrix.todense()

    #Build Y as described in Proj. 2 description
    Y_np = numpy_array[:,-1]
    numpy_array = numpy_array[:,:-1]
    rows, columns = numpy_array.shape

    unique_classes = len(np.unique([y[0] for y in Y_np.tolist()]))

    delta = np.zeros((unique_classes,rows))
    #for r in numpy_array[:,-1]:
    #    for c in range(0,rows):
    #        if(r==numpy_array[c,-1]):
    #            delta[r-1,c] =1
    for c in range(0, rows):
        delta[Y_np[c]-1,c] = 1

    #numpy_array2 = numpy_array[:,:-1]
    #rows, columns = numpy_array.shape
    ones = np.ones((rows,1))
    
    #X as described in Proj. 2 description
    numpy_array = numpy_array[:,1:]

    final_array = np.append(ones,numpy_array,1)
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
    return X_sparse, Y_sparse, delta_sparse, unique_classes, Y_np
    

#As written (the non-commented bits) assumes sparse matrix inputs
def grad_descent(X, Y, unique_classes, delta, lamb, learning_rate, iterations=None, limit=None):
    print("Start GD")
    X = X.toarray()
    rows, columns = X.shape
    ones_row = np.ones((rows, 1))
    print(X)
    print(ones_row)
    X = np.append(X, ones_row, axis=1)
    X = normalize(X, norm='l1', axis=0)
    X_t = X.transpose()
    W_sparse = sparse.rand(unique_classes,columns+1)
    X_t = csr_matrix(X_t)
    X = csr_matrix(X)


    Psparse = W_sparse.dot(X_t)
    Psparse = Psparse.todense()
    Psparse = np.exp(Psparse.data)
    Psparse[-1, :] = 1
    Psparse = normalize(Psparse, norm='l1', axis=0)
    print(Psparse)

    Psparse = lil_matrix(Psparse)
    Psparse = Psparse.tocsr()
    L_W = W_sparse.multiply(lamb * learning_rate)
    delta_P = ((delta-Psparse).dot(X))
    L_delta = delta_P.multiply(learning_rate)
    W_sparse = W_sparse + L_delta-L_W

    if iterations:
        for i in range(1,iterations):
            print("GD iter", i)

            Psparse = W_sparse.dot(X_t)
            Psparse = Psparse.todense()
            Psparse = np.exp(Psparse.data)
            Psparse[-1, :] = 1
            Psparse = normalize(Psparse, norm='l1', axis=0)
            #print(Psparse)

            Psparse = lil_matrix(Psparse)
            Psparse = Psparse.tocsr()
            L_W = W_sparse.multiply(lamb * learning_rate)
            delta_P = ((delta - Psparse).dot(X))
            L_delta = delta_P.multiply(learning_rate)
            W_sparse = W_sparse + L_delta - L_W

    elif limit:
        count = 1
        while True:
            Psparse = W_sparse.dot(X_t)
            Psparse = Psparse.todense()
            Psparse = np.exp(Psparse.data)
            Psparse[-1, :] = 1
            Psparse = normalize(Psparse, norm='l1', axis=0)
            #print(Psparse)

            Psparse = lil_matrix(Psparse)
            Psparse = Psparse.tocsr()
            L_W = W_sparse.multiply(lamb * learning_rate)
            delta_P = ((delta - Psparse).dot(X))
            L_delta = delta_P.multiply(learning_rate)

            W_next = W_sparse + L_delta - L_W
            diff = np.abs(csr_matrix.max(W_next) - csr_matrix.max(W_sparse))
            W_sparse = W_next

            print("GD iter", count, "- Diff =", diff)
            count += 1

            if diff <= limit:
                break

            

    print("Done w/ GD\n")
    #print(W_sparse.todense())
    #print(Psparse.todense())
    return W_sparse


def classify(file, W, K):
    """
    Classifies testing data using set of weights W. 
    Uses equations 27 and 28 from Mitchell Ch.3 but using sparse matrix multiplication.
    Parameters
    ----------
    file : str
        Filename of the test data
    W : csr_matrix
        Matrix of weights
    K : int
        Number of unique classes
    """
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
                X.append(1)
                X = csr_matrix(X)

                sum_mat = W_1.dot(X.transpose()).todense()
                sum_mat = np.exp(sum_mat)
                
                c = np.argmax(sum_mat) + 1
                ans = [int(example[0]), c]

                #Print answer pairs and write to csv
                print(ans)
                solout.writerow(ans)
                
def classify_conf(Xs, W, K):
    """
    Classifies testing data using set of weights W. 
    Uses equations 27 and 28 from Mitchell Ch.3 but using sparse matrix multiplication.
    Parameters
    ----------
    file : str
        Filename of the test data
    W : csr_matrix
        Matrix of weights
    K : int
        Number of unique classes
    """
    count = 1
    ans =[]
    W_1 = W[:,1:]
    print(W_1.shape,Xs.shape)

    for example in Xs:
        print("Classifying row", count)
        print(example.shape)
        count += 1
        X = example
        #X = [int(x) for x in X]
        #X.append(1)
        X = csr_matrix(X)
        print(X.shape)
        sum_mat = W_1.dot(X.transpose()).todense()
        sum_mat = np.exp(sum_mat)
                
        c = np.argmax(sum_mat) + 1
        ans.append(c)
    return ans
                #Print answer pairs and write to csv
                #print(ans)
                #solout.writerow(ans)

if __name__ == "__main__":
    cols_comp =[]
    
    #results = to_sparse_mat("/home/jared/Downloads/training.csv")
    
    #results = to_sparse_mat("smalltrain.csv")
    text_matrix = to_sparse_mat_for_conf('/home/jared/Downloads/training.csv',0, 6000)
    class_matrix = to_sparse_mat_for_conf('/home/jared/Downloads/training.csv',6001, 12000)
    cols = class_matrix[4]
    for i in cols:
        cols_comp.append(i.item())
    new_class_matrix = class_matrix[0]

    #W = grad_descent(results[0], results[1], results[3], results[2], .001, .001, iterations=1000)
    
    W = grad_descent(text_matrix[0], text_matrix[1], text_matrix[3], text_matrix[2], .001, .001, iterations=1000)

    #W = W[:,1:]
    
    #K = results[3] - 1
    
    spec_array = classify_conf(new_class_matrix, W, 20)
    actual = cols_comp
    predicted = spec_array
    disp_label = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    matrix = confusion_matrix(actual,predicted, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,18, 19, 20])

    print(matrix)
    
    disp = ConfusionMatrixDisplay(confusion_matrix= matrix, display_labels=disp_label)
    disp.plot()
    plt.show()
    
    #classify("testing.csv", W, K)
    
    #classify("smalltest.csv", W, K)