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
    print("1",numpy_array,"\n")
    Y = numpy_array[:,-1]
    print("2",Y,"\n")

    numpy_array = numpy_array[:,:-1]
    print("3",numpy_array,"\n")

    rows, columns = numpy_array.shape
    ones = np.ones((rows,1))
    print("ONES",ones,"\n")
    final_array = np.append(ones,numpy_array,1)
    print("4",final_array,"\n")


    