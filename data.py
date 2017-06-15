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
    y = np.zeros((nb_tests, seq_size), dtype="float32")

    for i in range(nb_tests):
        n = np.random.random_integers(vect_size, size=(seq_size,))-1
        for j in range(seq_size):
            x[i, j, n[j]] = 1.
            if n[j] in n[:j]:
                y[i, j] = 1.

    return x, y
