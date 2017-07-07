import plotly as py
import plotly.graph_objs as go
import numpy as np

from extractor import *

def convert_graph(mat):
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


def test_inclusion(t1, t2, n, stop_rate=0.1):
    """
    Generates n words over t2 and returns the porcentages of false/true positive/negative
    """
    
    walks = [rand_walk(t2) for i in range(n)]
    words = [w[1] for w in walks]
    prediction = [w[0] for w in walks]

    solution = [test_word(t1, w) for w in words]
    
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

      
def test_network(model, automaton, nb_tests,alphabet, batch_size, 
        nb_words, stop_rate, max_length):
    x = []
    y = []   
    print("generating the predictions")
    for i in range(3, max_length+1):
        x_in, _, _ = automaton_batch(nb_tests,alphabet, i, automaton=automaton)
        y_in = model.predict(x_in, batch_size = batch_size)
        x += [[str(a.tolist().index(1.)) for a in b] for b in x_in]
        y += y_in.tolist()
   
    tpr = []
    fpr = []

    step = 0.1
    for i in range(1, 10):
        threshold = step*i
        print("\nThreshold: ", threshold)
        x_in = ["".join(w) for w in x]
        l_plus = [x for i, x in enumerate(x_in) if y[i][0] > threshold]
        l_minus = [x for i, x in enumerate(x_in) if y[i][0] <= threshold]
    
        a2 = extract(l_plus, l_minus)    
        a = convert_graph(automaton)

        r = test_inclusion(a, a2, nb_words, stop_rate=stop_rate)
        print("True positive rate:  ", r[0])
        print("False positive rate: ", r[1])
        print("True negative rate:  ", r[2])
        print("False negative rate: ", r[3])
        tpr += [r[0]]
        fpr += [r[1]]

    return tpr, fpr

def trace_ROC(tpr, fpr): 
    trace = go.Scatter(
            x = np.array(fpr),
            y = np.array(tpr),
            name = "Memory",
            mode = 'markers',
            marker = dict(
                size = 10,
                color = 'rgba(152, 0, 0, .8)'))

    plots = [trace]

    layout = dict(title = 'Styled Scatter',
              yaxis = dict(zeroline = False, range=[0, 1]),
              xaxis = dict(zeroline = False, range=[0, 1])
             )

    fig = dict(data=plots, layout=layout)    
    py.offline.plot(fig, filename='roc.html')
 


