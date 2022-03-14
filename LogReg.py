#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 23:08:51 2022

"""
from numpy import genfromtxt
import numpy as np
import csv
import statistics as st

def to_sparse_mat(file):
    csv = np.genfromtxt (file, delimiter=",")
    numpy_array = csv[:,1:]
    #Build Y as described in Proj. 2 description
    Y = numpy_array[:,-1]
    numpy_array = numpy_array[:,:-1]
    rows, columns = numpy_array.shape
    ones = np.ones((rows,1))
    #X as described in Proj. 2 description
    final_array = np.append(ones,numpy_array,1)
    #k as described in Proj. 2 description
    unique_classes = len(np.unique(Y))
    #init delta as described in Proj. 2 description (k x m matrix of zeros)
    delta=np.zeros((unique_classes,columns))

    