import math
import numpy as np
import csv
from scipy.sparse import csr_matrix
import datetime
from collections import defaultdict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pprint import pprint
import matplotlib.pyplot as plt
from scipy.stats import entropy
import array as arr

def my_func(array_O_words,words_txt,MAPS):
    """
    

    Parameters
    ----------
    array_O_words : Array
        Array representing the original data set including the true classes.
    words_txt : path to vocabulary.txt file
        .txt file holding ordered set of vocabulary.
    MAPS : List
        List of calculated MAP approximations of P(X_i|Y).
    

    Returns
    -------
    None.

    """
    print("Start")
    ans =[]
    #ordered words from vocabulary.txt
    words = np.loadtxt(words_txt, usecols=0, skiprows=1, dtype='str')
    Ys = []
    #num_of_rows = []
    Hs =[]
    counts = arr.array('i', [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    #exampleCount = 0
    for col in range(1,61189):
        for row in range(0,12000):
            #If x_i for example j >0, find class it belongs to, increment index
            #equivalent to (class # -1) by 1 to have number for each count given 
            #this x_i
            if array_O_words[row][col] >0:
                counts[int(array_O_words[row][-1])-1] = counts[int(array_O_words[row][-1])-1]+1
        A = counts.tolist()
        #get probabilities of classes dividing each count by sum of class counts
        A = [x/(sum(A)) if sum(A) >0 else x for x in A]
        Ys.append(A)
        for i in range(0,20):
            counts[i] =0
    print(np.shape(Ys))
    for i in range(0,len(Ys)):
        Hs.append(entropy(Ys[i],base =2))
    print(np.shape(Hs),np.shape(MAPS))
    Hs = [-1*Hs[i]*MAPS[i] for i in range(0,len(Hs))]
    #find max, append to ans, delete that index, repeat to get the top 100.
    Hs =arr.array('i').fromlist(words.tolist())
    for i in range(0,100):
        ans.append(words[Hs.index(max(Hs))])
        Hs[Hs.index(max(Hs))] = -1000000
    print(ans)
    """
    Pseudo-code
    -Entropy(P(Y|x_i)) * MAP est
    """

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
    input probs_calc:
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
                    #Matrix multiplication possibly?
                    ## csr_A.multiply(csr_B) will do matrix multiplication between
                    ##the two sparse matrices.
                    Map_calc = Map_calc + row[idxrow] * int(el)
            Map_calc = Map_calc + p_v[row_count]
            row_count = row_count + 1
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
    input probs_calc: array of
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
                        Map_calc = Map_calc + row[idxrow] * int(el)
                Map_calc = Map_calc + p_v[row_count]
                row_count = row_count + 1
                if Map_calc > max_prob:
                    max_prob = Map_calc
                    max_idx = idx + 1
                Map_calc = 0
            row_count = 0
            file_object.write(str(testid) + ","+str(max_idx)+'\n')
            max_prob = -10000000
            max_idx = 0

Vocabulary = 61188
#beta = 1/Vocabulary
beta= .0015
#cols_comp = []
class_wrong = defaultdict(int)
#array for MLE P(Yk) for each class 1 x 20
p_v = []
#array for MaP for P(X|Y) for each class Size 20 x 61188
prob_calcs = []
text_matrix, total_rows = csv_to_sparse('/home/jared/Downloads/training.csv', 61190)
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
    filtered_matrix_list = [x - 1 for x in filtered_matrix]
    num_rows = len(filtered_matrix_list)
    probs = math.log2(num_rows/total_rows)
    p_v.append(probs)
    man_data = text_matrix[filtered_matrix_list, :].sum(axis=0).tolist()[0]
    del man_data[0]
    del man_data[-1]
    total_words = sum(man_data)
    #used code from https://stackoverflow.com/questions/54629298/how-to-use-vectorized-numpy-operations-on-lambda-functions-that-return-constant
    func = lambda x: math.log2((x + beta)/(total_words + 1))
    vfunc = np.vectorize(func)
    prob_bayes = vfunc(man_data)
    prob_calcs.append(prob_bayes)
    
my_func(text_matrix.toarray(), '/home/jared/Downloads/vocabulary.txt', prob_bayes)
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
classify('/home/jared/Downloads/testing.csv', p_v, prob_calcs)
