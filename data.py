import numpy as np

def parity_batch(nb_tests, upper_bound):
    nbs = np.random.random_integers(upper_bound, size=(nb_tests,))-1
    x = np.zeros((nb_tests, upper_bound), dtype="float32")
    y = np.zeros((nb_tests), dtype="float32")
    for i, n in enumerate(nbs):
        x[i][n] = 1.
        if n%2==0:
            y[i] = 1.

    return x, y
