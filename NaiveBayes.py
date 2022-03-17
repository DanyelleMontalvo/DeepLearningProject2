import math
import numpy as np
import csv
from scipy.sparse import csr_matrix

def csv_to_sparse(csvdoc, colnum):
    col_count = []
    row_count = []
    data_count = []
    rowcount = 0
    #code incorporated from from https://docs.python.org/3/library/csv.html
    with open(csvdoc, newline='', encoding='utf-8-sig') as csvfile:
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
    return new_matrix, rowcount

def classify(csvdoc, p_v, probs_calc):
    testid = 0
    max_prob = -100000
    max_idx = 0
    Map_calc = 0
    row_count = 0
    file_object = open('sample.csv', 'w+')
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
                        #Matrix multiplication possibly?
                        ## csr_A.multiply(csr_B) will do matrix multiplication between
                        ##the two sparse matrices. 
                        Map_calc = Map_calc + row[idxrow] * int(el)
                Map_calc = Map_calc + p_v[row_count]
                print(Map_calc)
                row_count = row_count + 1
                if Map_calc > max_prob:
                    max_prob = Map_calc
                    max_idx = idx + 1
                Map_calc = 0
            row_count = 0
            file_object.write(str(testid) + ","+str(max_idx)+'\n')
            max_prob = -10000
            max_idx = 0


Vocabulary = 61188
beta = 1/Vocabulary
#array for MLE P(Yk) for each class 1 x 20
p_v = []
#array for MaP for P(X|Y) for each class Size 20 x 61188
prob_calcs = []
text_matrix, total_rows = csv_to_sparse('training.csv', 61190)
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
    print(prob_bayes)
    prob_calcs.append(prob_bayes)




classify('testing 3.csv', p_v,prob_calcs)



