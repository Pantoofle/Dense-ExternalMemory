import numpy as np
import sys

sys.path.append("layers/")

from keras.callbacks import *
from builder import *
from data import *

VECTOR_SIZE=10
MEMORY_SIZE=3
SEQ_LENGTH=7
INCLUDE_PROB=0.5

NB_TRAIN=5000
NB_TESTS=100

SAVE_DIR="models/"

if __name__ == "__main__":
    print("Starting process...")

    if(len(sys.argv) > 2 and sys.argv[1] == "load"):
        print("Loading model...")
        model = load_model(SAVE_DIR+sys.argv[2], 
                {'IO_Layer': IO_Layer})
        save_name = sys.argv[2]
    else:
        print("Building new model...")
        model = build_RNN(input_shape=(SEQ_LENGTH, VECTOR_SIZE), 
                memory_size=MEMORY_SIZE, 
                vect_size=VECTOR_SIZE, 
                output_size=1)
        save_name = input("Enter model name: ")

    print("Compiling...")
    model.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    
    model.summary() 
    
    print("Input shape: ", model.input_shape)
    print("Getting data...")
    x_in, y_in = include_batch(NB_TRAIN, 
            SEQ_LENGTH, 
            VECTOR_SIZE)

    print("x: ", x_in.shape)
    print("y: ", y_in.shape)

    print("Saving model...")
    model.save(SAVE_DIR+save_name)
    print("Training...")
    model.fit(x_in, y_in) 

    print("Saving model...")
    model.save(SAVE_DIR+save_name)

    print("Testing model...")
    x_in, y_in = include_batch(NB_TESTS, 
            SEQ_LENGTH, 
            VECTOR_SIZE)  
    
    pred = model.predict(x_in)

    pred = [p > 0.5 for p in pred]
    res = [p > 0.5 for p in y_in]
    ok = sum([1 for x, y in zip(pred, res) if x == y  ])
    print("Nb tests:   ", NB_TESTS)
    print("Nb success: ", ok)

