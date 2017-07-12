import numpy as np
import re

from extractor import *
from data import *

def m2dic_convert(mat):
    """
    Convert a graph represented by a matrix: mat[i][j] is the list of characters that read from i lead into j
    Into a dictionnary graph: 
        - t["entry"]["id"][0] is the name of the entry node
        - t[i] contains the informations about node i
        - t[i]["exit"][0] = True if i is an exit node, else, False
        - t[i][c] is the list of the nodes reached after reading character c from i.
            if deterministic, len(t[i][c]) = 1
            else, we may have len(t[i][c]) > 1
    """
    n = len(mat)
    t = {"entry": {"id": ["0"]}}
    
    for i in range(n):
        t[str(i)] = {"exit": [False]}
        for j in range(n):
            for c in mat[i][j]:
                t[str(i)][str(c)] = [str(j)] 
    
    t[str(n-1)]["exit"] = [True]

    return t
 
def dot2dic_convert(dot):
    """
    converts a graph encoded in its dot form into the dictionnary
    """
    
    t = {"entry": {"id": [0]}}

    # Extracting exit nodes
    exit = []
    m = re.search(r'.*doublecircle.*; (.*);', dot)
    if m is not None:
        ex_txt = m.group(1)
        it = re.finditer(r'\d+', ex_txt, re.S)
        for i in it:
            exit.append(int(ex_txt[i.start(): i.end()]))

    # Extracting links
    transition = []
    it = re.finditer(r'\t(.*) -> (.*);', dot)
    for i in it:
        link = dot[i.start():i.end()]
        it2 = re.finditer(r'\d+', link)
        for j in it2:
            transition.append(int(link[j.start():j.end()]))

    transition =transition[2:]

    def build_if_not(t, n):
        if n in t:
            return
        else:
            t[n] = {"exit": [False]}
            return

    # Building graph
    for i in range(0,len(transition),3):
        dep = transition[i]
        to  = transition[i+1]
        label = str(transition[i+2])
        
        build_if_not(t, dep)
        build_if_not(t, to)
        
        t[dep][label] = [to]

    # Setting the exit nodes
    for i in exit:
        t[i]["exit"][0] = True

    return t
        

def test_inclusion(t1, t2, x):
    """
    Tests the words x over t2 and returns the ratio of false/true positive/negative
    """
    
    prediction = [test_word(t2, w) for w in x]

    solution = [test_word(t1, w) for w in x]
    
    n = len(x)
    
    npos = len([1 for p in solution if p])
    nneg = len([1 for p in solution if not p])
    
    if npos == 0:
        tpr = 1.
        fnr = 1.
    else:
        true_pos = len([1 for i in range(n) if solution[i] and prediction[i]])
        tpr = true_pos*1./npos

        false_neg = len([1 for i in range(n) if solution[i] and (not prediction[i])])
        fnr = false_neg*1./npos
    if nneg == 0:
        tnr = 1.
        fpr = 1.
    else:
        true_neg = len([1 for i in range(n) if (not solution[i]) and (not prediction[i])])
        tnr = true_neg*1./nneg

        false_pos = len([1 for i in range(n) if (not solution[i]) and prediction[i]])
        fpr = false_pos*1./nneg


    return tpr, fpr, tnr, fnr

      
def test_network(model, automaton, alphabet, batch_size, nb_words, min_length, max_length):
    """
    Generates a test set of words over the automaton
    Returns the different rates (true positives and false positives)
    computed with a lot of different thresholds
    """
    
    x = []
    y = []   
    print("Generating the words and predictions")
    for i in range(min_length, max_length+1):
        x_in, _, _ = automaton_batch(nb_words, alphabet, i, automaton=automaton)
        y_in = model.predict(x_in, batch_size = batch_size)
        x += [[str(a.tolist().index(1.)) for a in b] for b in x_in]
        y += y_in.tolist()
    
    y = [w[0] for w in y]
    print(len(y), " values")
    
    thresholds = sorted(list(set(y)))
    print(len(thresholds), " different")
   
    tpr = []
    fpr = []
    i = 1
    for t in thresholds:
        print("Threshold ", i, "/", len(thresholds), ": ", t)
        i += 1

        a = alf_infere(model, automaton, t, alphabet) 
        a = dot2dic_convert(a)

        r = test_inclusion(a, automaton, x)
        #  print("True positive rate:  ", r[0])
        #  print("False positive rate: ", r[1])
        #  print("True negative rate:  ", r[2])
        #  print("False negative rate: ", r[3])
        tpr += [r[0]]
        fpr += [r[1]]

    return tpr, fpr

