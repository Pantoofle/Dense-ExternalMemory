import numpy as np
import random 
import graphviz as gv

from keras import *
from keras.models import *
from keras.models import model_from_json

from extractor import *

def parity_batch(nb_tests, upper_bound):
    """
    Generates a set of vectors encoding integers with "one-hot" method.
    Solution is to answer wether each number is odd or even
    """

    nbs = np.random.random_integers(upper_bound, size=(nb_tests,))-1
    x = np.zeros((nb_tests, upper_bound), dtype="float32")
    y = np.zeros((nb_tests), dtype="float32")
    for i, n in enumerate(nbs):
        x[i][n] = 1.
        if n%2==0:
            y[i] = 1.

    return x, y

def include_batch(nb_tests, seq_size, vect_size):
    """
    Generates sequences of vectors encoded with "one-hot" method
    At each step, answer is to say if actual vector appeared earlier in the sequence
    """
    
    x = np.zeros((nb_tests, seq_size, vect_size), dtype="float32")
    y = np.zeros((nb_tests, seq_size, 1), dtype="float32")

    for i in range(nb_tests):
        n = np.random.random_integers(vect_size, size=(seq_size,))-1
        for j in range(seq_size):
            x[i, j, n[j]] = 1.
            if n[j] in n[:j]:
                y[i, j, 0] = 1.

    return x, y


def s (n):
    """
    Computes the number os steps needed to reach 1 when Syraccuse function is applied 
    to the entry n
    """
    if n == 1:
        return 0
    elif n % 2 == 0:
        return 1 + s(n/2)
    else:
        return 1 + s(3*n + 1)

def f(n):
    """
    Applies Syraccuse function to n, returns the next value
    """
    if n == 1:
        return 1
    elif n % 2 == 0:
        return n/2
    else:
        return 3*n + 1

def syracuse_batch(nb_tests, entry_size, scalar):
    """
    Encodes a set of numbers in binary
    Solution is to give the number of times one must apply the Syraccuse function before 
    reaching 1

    VERY HARD for a network
    """
    n_dec = np.random.random_integers(2**(entry_size-1)+1, size=(nb_tests,))
    
    r_dec = np.array([s(i) for i in n_dec])
    wait = int(max(r_dec)*scalar)

    r_bin = np.array(["{0:b}".format(n) for n in r_dec])
    r_len = max([len(r) for r in r_bin])
    l = max(r_len, entry_size)

    n_bin = np.array(["{0:b}".format(n).zfill(l) for n in n_dec])
    n = np.array([[int(c) for c in s] for s in n_bin])
    
    r_bin = np.array(["{0:b}".format(n).zfill(l) for n in r_dec])
    r = np.array([[int(c) for c in s] for s in r_bin], dtype="float32")

    x = np.zeros((nb_tests, wait, l),dtype="float32")
    for i, e in enumerate(n):
        x[i, 0] = e

    return wait, l, x, r

def generate_automaton(states, alphabet, path=None):
    """
    Generates a random automaton with given number of states and given alphabet size
    If path is given, will use graphviz to trace it and render it
    Returs it under the form of a matrix of transition
    """
    mat = [[[] for _ in range(states)] for _ in range(states)]
    
    for source in range(states):
        for letter in range(alphabet):
            dest = random.choice(range(states))
            mat[source][dest] += [letter]

    if path is not None:
        graph = gv.Digraph(format="svg")
        for i in range(states):
            graph.node(str(i))
            for j in range(states):
                for l in mat[i][j]:
                    graph.edge(str(i), str(j), str(l))
        graph.render(path)
    
    return mat


def automaton_batch(nb_tests, alphabet, length, automaton=None, states=0):
    """
    Creates a set of words, as sequence of letters, on a given automaton.
    Answer is to say if given word is accepted by the automaton
    If no automaton is given, a new one will be generated
    """
    
    if automaton is None:
        automaton = generate_automaton(states, alphabet)
        automaton = m2dic_convert(automaton)

    x = np.zeros((nb_tests, length, alphabet))
    y = np.zeros((nb_tests, 1))

    tot = 0

    for i in range(nb_tests):
        ok, path = rand_walk(automaton, length=length )
        for j in range(length):
            x[i][j][int(path[j])] = 1.

        if ok:
            y[i] = 1.
            tot +=1

    return x, y, tot

    



