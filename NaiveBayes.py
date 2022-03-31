import math
import numpy as np
import csv
from scipy.sparse import csr_matrix
import datetime
from collections import defaultdict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pprint import pprint
import matplotlib.pyplot as plt

def csv_to_sparse(csvdoc, colnum, skipstart=0, stopafter=0):
    """
    Function to turn csv into a sparse matrix
    input csvdoc: csv file
    output: returns the sparse Matrix and a rowcount
    """
    col_count = []
    row_count = []
    data_count = []
    rowcount = 0
    #code incorporated from from https://docs.python.org/3/library/csv.html
    #This code helps create a csr matrix from a csv file
    with open(csvdoc, newline='', encoding='utf-8-sig') as csvfile:
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
    return new_matrix, rowcount

def dropcols_coo(M, idx_to_drop):
    """
    Function that drops an index from the csr matrix
    Taken from the website
    https://stackoverflow.com/questions/23966923/delete-columns-of-matrix-of-csr-format-in-python
    """
    idx_to_drop = np.unique(idx_to_drop)
    C = M.tocoo()
    keep = ~np.in1d(C.col, idx_to_drop)
    C.data, C.row, C.col = C.data[keep], C.row[keep], C.col[keep]
    C.col -= idx_to_drop.searchsorted(C.col)    # decrement column indices
    C._shape = (C.shape[0], C.shape[1] - len(idx_to_drop))
    return C.tocsr()

def classify_conf(class_matrix, p_v, probs_calc):
    """
    Function to classify an input specifically for the confusion matrix
    input csvdoc: an input file that contains data to be classified
    input p_v: The array of probabilities
    input probs_calc: array of probabilities
    output a classified array for the confusion matrix
    """
    max_prob = -1000000
    max_idx = 0
    Map_calc = 0
    row_count = 0
    array_classified = []
#    for i in range(0, 2400):
    for i in range(0, 2400):
        row_mat = class_matrix.getrow(i)
        row_dense = row_mat.toarray()
        row_dense = list(row_dense)
        row_dense = row_dense[0]
        testid = row_dense[0]
        row_dense = row_dense[1:]
        for idx, row in enumerate(probs_calc):
            for idxrow, el in enumerate(row_dense):
                if el == 0:
                    continue
                else:
                    Map_calc = Map_calc + row[idxrow] * int(el)
            Map_calc = Map_calc + p_v[row_count]
            row_count = row_count + 1
            #Find the maximum probability
            if Map_calc > max_prob:
                max_prob = Map_calc
                max_idx = idx + 1
            Map_calc = 0
        row_count = 0
        array_classified.append(max_idx)
        #file_object.write(str(testid) + ","+str(max_idx)+'\n')
        max_prob = -100000
        max_idx = 0
    return array_classified

def classify(csvdoc, p_v, probs_calc):
    """
    Function to classify an input
    input csvdoc: an input file that contains data to be classified
    input p_v: The array of probabilities
    input probs_calc: array of probabilities
    output is a csv file called classified.csv that has
    """
    testid = 0
    max_prob = -10000000
    max_idx = 0
    Map_calc = 0
    row_count = 0
    file_object = open('classified.csv', 'w+')
    file_object.write("id,class\n")
    with open(csvdoc, newline='', encoding='utf-8-sig') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for training_row in csvreader:
            testid = training_row[0]
            del training_row[0]
            for idx, row in enumerate(probs_calc):
                for idxrow, el in enumerate(training_row):

                    if int(el) == 0:
                        continue
                    else:
                        #calculate probabilities for each feature
                        Map_calc = Map_calc + row[idxrow] * int(el)
                #Calulate the maximum liklihood estimate
                Map_calc = Map_calc + p_v[row_count]
                row_count = row_count + 1
                #find the class that provides the highest probability
                if Map_calc > max_prob:
                    max_prob = Map_calc
                    max_idx = idx + 1
                Map_calc = 0
            row_count = 0
            #write max class to file
            file_object.write(str(testid) + ","+str(max_idx)+'\n')
            max_prob = -10000000
            max_idx = 0

Vocabulary = 61188
beta = 1/Vocabulary
#beta= .0015
#cols_comp = []
class_wrong = defaultdict(int)
#array for MLE P(Yk) for each class 1 x 20
p_v = []
#array for MaP for P(X|Y) for each class Size 20 x 61188
prob_calcs = []
text_matrix, total_rows = csv_to_sparse('training.csv', 61190)
#These are used for the confusion matrix calculations
#text_matrix, total_rows= csv_to_sparse('training.csv', 61190, 0, 9600)
#class_matrix, total_rows_class= csv_to_sparse('training.csv', 61190, 9601, 12000)
#cols = class_matrix.getcol(61189)
#cols = cols.toarray()
#for i in cols:
#    cols_comp.append(i.item())

#new_class_matrix = dropcols_coo(class_matrix, 61189)
for classnum in range(1, 21):
    filtered_matrix = text_matrix[text_matrix[:, 61189] == classnum].tolist()[0]
    #code from https://stackoverflow.com/questions/4918425/subtract-a-value-from-every-number-in-a-list-in-python
    #Subtract the key ids from the data
    filtered_matrix_list = [x - 1 for x in filtered_matrix]
    num_rows = len(filtered_matrix_list)
    #Take logrithm of the probabilities for future calculations
    probs = math.log2(num_rows/total_rows)
    p_v.append(probs)
    man_data = text_matrix[filtered_matrix_list, :].sum(axis=0).tolist()[0]
    del man_data[0]
    del man_data[-1]
    total_words = sum(man_data)
    #used code from https://stackoverflow.com/questions/54629298/how-to-use-vectorized-numpy-operations-on-lambda-functions-that-return-constant
    #Calculate the probabilities for each class
    func = lambda x: math.log2((x + beta)/(total_words + 1))
    vfunc = np.vectorize(func)
    prob_bayes = vfunc(man_data)
    #put probabilities in a 1x20 matrix
    prob_calcs.append(prob_bayes)
#This section is used to create the confusion matrix but is commented out upon submission
#spec_array = classify_conf(new_class_matrix, p_v, prob_calcs)
#actual = cols_comp
#predicted = spec_array
#disp_label = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
#matrix = confusion_matrix(actual,predicted, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,18, 19, 20])
#print(matrix)
#disp = ConfusionMatrixDisplay(confusion_matrix= matrix, display_labels=disp_label)
#disp.plot()
#plt.show()
classify('testing.csv', p_v, prob_calcs)



