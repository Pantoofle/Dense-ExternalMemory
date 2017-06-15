import numpy as np
import sys

sys.path.append("layers/")

from keras.callbacks import *
from builder import *
from data import *

VECTOR_SIZE=10
MEMORY_SIZE=20
SEQ_LENGTH=5
INCLUDE_PROB=0.5

NB_TRAIN=5000
NB_TESTS=100

SAVE_DIR="models/"

if __name__ == "__main__":
    print("Starting process...")

    if(len(sys.argv) > 2 and sys.argv[1] == "load"):
        print("Loading model...")
        model = load_model(SAVE_DIR+sys.argv[2], {'IO_Layer': IO_Layer})
        save_name = sys.argv[2]
    else:
        print("Building new model...")
        model = build_RNN(input_shape=(VECTOR_SIZE+1,), 
                memory_size=MEMORY_SIZE, 
                vect_size=VECTOR_SIZE, 
                output_size=1)
        save_name = input("Enter model name: ")

    print("Compiling...")
    model.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    
    model.summary() 
    empty_mem = model.layers[1].get_weights()
    
    print("Input shape: ", model.input_shape)
    print("Getting data...")
    x_in, y_in = include_batch(NB_TRAIN, 
            SEQ_LENGTH, 
            VECTOR_SIZE, 
            INCLUDE_PROB)  
 
    print("Saving model...")
    model.save(SAVE_DIR+save_name)
    print("Training...")

    i = 1
    for x_seq, y_seq in zip(x_in, y_in):
        print("Training sequence ", i,"/", NB_TRAIN)
        i += 1
        model.layers[1].set_weights(empty_mem)
        previous = np.zeros(shape=(1, 1), dtype="float32")
        for x in x_seq[:-1] :
            x = np.reshape(x, (1, VECTOR_SIZE))
            v = np.concatenate((previous, x), axis=1)
            previous = model.predict(v) 
        x = np.reshape(x_seq[-1], (1, VECTOR_SIZE))
        v = np.concatenate((previous, x), axis=1)
        solution = np.array([[y_seq]])
        model.train_on_batch(v, solution)
    print("Training done !")

    print("Saving model...")
    model.save(SAVE_DIR+save_name)

    print("Testing model...")
    x_in, y_in = include_batch(NB_TESTS, 
            SEQ_LENGTH, 
            VECTOR_SIZE, 
            INCLUDE_PROB)  
    
    pred = np.zeros((NB_TESTS, 1), dtype="float32")
    i=0
    for x_seq, y_seq in zip(x_in, y_in):
        print("Testing sequence ", i,"/", NB_TESTS)
        i += 1
        previous = np.zeros(shape=(1, 1), dtype="float32")
        for x in x_seq :
            x = np.reshape(x, (1, VECTOR_SIZE))
            v = np.concatenate((previous, x), axis=1)
            previous = model.predict(v) 
        x = np.reshape(x_seq[-1], (1, VECTOR_SIZE))
        v = np.concatenate((previous, x), axis=1)
        pred[i-1] = previous[0, 0]
 
    pred = [p > 0.5 for p in pred]
    res = [p > 0.5 for p in y_in]
    ok = sum([1 for x, y in zip(pred, res) if x == y  ])
    print("Nb tests:   ", NB_TESTS)
    print("Nb success: ", ok)

