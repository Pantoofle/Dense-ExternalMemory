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

def include_batch(nb_tests, seq_size, vect_size, include_prob):
    x = np.zeros((nb_tests, seq_size+2, vect_size), dtype="float32")
    y = np.zeros(nb_tests, dtype="float32")

    for k in range(nb_tests):
        seq = np.random.random((seq_size, vect_size))
    
        # Adding delimiter
        seq = np.append(seq, np.zeros((1, vect_size), dtype="float32"), axis=0)

        # Building question and solution
        if np.random.random() < include_prob:
            # Choosing the added vector
            i = np.random.randint(seq_size)
            seq = np.append(seq, [seq[i]], axis=0)
            sol = 1.
        else:
            seq = np.append(seq, np.random.random((1, vect_size)), axis=0)
            sol = 0.

        x[k] = seq
        y[k] = sol

    return x, y
