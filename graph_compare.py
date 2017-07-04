import numpy as np

from extractor import *

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

    nb_1_and_2 = sum([int(test_word(t2, w[1])) for w in w1 if w[0]])/(len([1 for w in w1 if w[0]])+1)
    nb_2_and_1 = sum([int(test_word(t1, w[1])) for w in w2 if w[0]])/(len([1 for w in w2 if w[0]])+1)
    nb_no1_and_no2 = sum([int(not test_word(t2, w[1])) for w in w1 if not w[0]])/(len([1 for w in w1 if not w[0]])+1)
    nb_no2_and_no1 = sum([int(not test_word(t1, w[1])) for w in w2 if not w[0]])/(len([1 for w in w2 if not w[0]])+1)

    return nb_1_and_2, nb_2_and_1, nb_no1_and_no2, nb_no2_and_no1
