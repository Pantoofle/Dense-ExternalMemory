import numpy as np

from extractor import *
from data import *

def rand_walk(t, stop_rate=0.1):
    """
    Generate a word over t
    It may or may not end in a final state
    After each step, the walk has probability stop_rate to just end there
    """
    
    word = []
    state = t["entry"]["id"][0]

    while True:
        possib = [c for c in t[state] if c != "exit"]
        if len(possib) == 0:
            #  print("No place to go...")
            break
        i = np.random.randint(len(possib))
        c = possib[i]
        #  print("Next move: ", c)

        word += [c]

        state = t[state][c][0]
        
        if stop_rate > np.random.random():
            #  print("The random said... STOP")
            break

    return t[state]["exit"][0], word

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
    Generates n words over t1 and t2 and returns the porcentages of inclusion
    of t1 into t2 and t2 into t1
    """
    
    w1 = [rand_walk(t1) for i in range(n)]
    w2 = [rand_walk(t2) for i in range(n)]
    
    l1 = len([1 for w in w1 if w[0]])
    l2 = len([1 for w in w2 if w[0]])
    l3 = len([1 for w in w1 if not w[0]])
    l4 = len([1 for w in w2 if not w[0]])
    
    if l1 == 0:
        nb_1_and_2 = 1.
    else:
        nb_1_and_2 = sum([int(test_word(t2, w[1])) for w in w1 if w[0]])/l1

    if l2 == 0:
        nb_2_and_1 = 1.
    else:
        nb_2_and_1 = sum([int(test_word(t1, w[1])) for w in w2 if w[0]])/l2

    if l3 == 0:
        nb_no1_and_no2 = 1.
    else: 
        nb_no1_and_no2 = sum([int(not test_word(t2, w[1])) for w in w1 if not w[0]])/l3

    if l4 == 0:
        nb_no2_and_no1 = 1.
    else:
        nb_no2_and_no1 = sum([int(not test_word(t1, w[1])) for w in w2 if not w[0]])/l4


    return nb_1_and_2, nb_2_and_1, nb_no1_and_no2, nb_no2_and_no1
      
def test_network(model, automaton, nb_tests, states, alphabet, batch_size, 
        nb_words, stop_rate, max_length):
    x = []
    y = []   
    print("generating the predictions")
    for i in range(3, max_length+1):
        x_in, _, _ = automaton_batch(nb_tests, states, alphabet, i, automaton=automaton)
        y_in = model.predict(x_in, batch_size = batch_size)
        x += [[str(a.tolist().index(1.)) for a in b] for b in x_in]
        y += y_in.tolist()
        
    x_in = ["".join(w) for w in x]
    l_plus = [x for i, x in enumerate(x_in) if y[i][0] > 0.5]
    l_minus = [x for i, x in enumerate(x_in) if y[i][0] <= 0.5]
    
    print("l_plus: ", len(l_plus))
    print("l_minus: ", len(l_minus))
    print("l_plus: ", l_plus[:10])
    print("l_minus: ", l_minus[:10])

    print("extracting the infered model from the predictions")
    a2 = extract(l_plus, l_minus)    
    a = convert_graph(automaton)

    print("testing inclusion")
    r = test_inclusion(a, a2, nb_words, stop_rate=stop_rate)
    print("a = original automaton")
    print("b = infered automaton")
    print("prop of words valid in a that are valid in b:     ", r[0])
    print("prop of words valid in b that are valid in a:     ", r[1])
    print("prop of words invalid in a that are invalid in b: ", r[2])
    print("prop of words invalid in b that are invalid in a: ", r[3])

