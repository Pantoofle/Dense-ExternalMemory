import numpy as np
from keras import *
from keras.models import *
from keras.models import model_from_json

def parity_batch(nb_tests, upper_bound):
    nbs = np.random.random_integers(upper_bound, size=(nb_tests,))-1
    x = np.zeros((nb_tests, upper_bound), dtype="float32")
    y = np.zeros((nb_tests), dtype="float32")
    for i, n in enumerate(nbs):
        x[i][n] = 1.
        if n%2==0:
            y[i] = 1.

    return x, y

def include_batch(nb_tests, seq_size, vect_size):
    x = np.zeros((nb_tests, seq_size, vect_size), dtype="float32")
    y = np.zeros((nb_tests, seq_size, 1), dtype="float32")

    for i in range(nb_tests):
        n = np.random.random_integers(vect_size, size=(seq_size,))-1
        for j in range(seq_size):
            x[i, j, n[j]] = 1.
            if n[j] in n[:j]:
                y[i, j, 0] = 1.

    return x, y


def memory_batch(nb_tests, seq_size, vect_size, memory_size):
    x = np.zeros((nb_tests, seq_size, vect_size), dtype="float32")
    y = np.zeros((nb_tests, seq_size, vect_size+memory_size+1), 
            dtype="float32")
    for i in range(nb_tests):
        n = np.random.random_integers(vect_size, size=(seq_size,))-1
        for j in range(seq_size):
            x[i, j, n[j]] = 1.
            y[i, j, n[j]+vect_size] = 1.
            y[i, j, n[j]] = 1.
    return x, y


def s (n):
    if n == 1:
        return 0
    elif n % 2 == 0:
        return 1 + s(n/2)
    else:
        return 1 + s(3*n + 1)

def f(n):
    if n == 1:
        return 1
    elif n % 2 == 0:
        return n/2
    else:
        return 3*n + 1

def syracuse_batch(nb_tests, entry_size, scalar):
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

def count_batch(nb_tests, seq_len):
    return None
