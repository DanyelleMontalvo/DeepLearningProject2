#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 23:08:51 2022

"""
import numpy as np
import csv
import statistics as st

def to_sparse_mat(file):
    file = open("sample.csv")
    numpy_array = np.loadtxt(file, delimiter=",")
    rows, columns = numpy_array.shape
    ones = np.ones(rows)
    final_array = np.append(ones,numpy_array,1)

def sigma(y_l,y_j):
    if y_l == y_j:
        return 1
    else:
        return 0
    