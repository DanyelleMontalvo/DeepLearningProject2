from scipy.stats import entropy
import array as arr
import numpy as np

def my_func(array_O_words,words_txt, probs_calc,MAPS):
    """
    

    Parameters
    ----------
    array_O_words : Array
        Array representing the original data set including the true classes.
    words_txt : .txt file
        .txt file holding ordered set of vocabulary.
    probs_calc : TYPE
        DESCRIPTION.
    MAPS : List
        List of calculated MAP approximations of P(X_i|Y).
    

    Returns
    -------
    None.

    """
    print("Start")
    ans =[]
    #ordered words from vocabulary.txt
    words = np.loadtxt("example3.txt", usecols=1, skiprows=1, dtype='str')
    Ys = []
    num_of_rows = []
    Hs =[]
    MAPS = []
    counts = arr.array('i', [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    #exampleCount = 0
    for col in range(1,61189):
        for row in range(0,len(num_of_rows)):
            #If x_i for example j >0, find class it belongs to, increment index
            #equivalent to (class # -1) by 1 to have number for each count given 
            #this x_i
            if array_O_words[row][col] >0:
                counts[int(array_O_words[row][-1])-1] = counts[int(array_O_words[row][-1])-1]+1
        A = counts.tolist()
        #get probabilities of classes dividing each count by sum of class counts
        A = [x/(sum(A)) for x in A]
        #61189x20
        Ys.append(A)
        counts[0:19] =0
    for i in range(0,61189):
        Hs.append(entropy(Ys[i]),base =2)
    
    Hs = [-1*Hs[i]*MAPS[i] for i in range(0,len(Hs))]
    #find max,
    for i in range(0,100):
        ans.append(words[Hs.index(Hs.max)])
        del words[Hs.index(Hs.max)]
    print(ans)
    """
    Pseudo-code
    -Entropy(P(Y|x_i)) * MAP est
    """