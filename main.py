import numpy as np
import sys

sys.path.append("layers/")

from keras.callbacks import *
from builder import *
from data import *

VECTOR_SIZE=50
MEMORY_SIZE=150
SEQ_LENGTH=50
INCLUDE_PROB=0.5

NB_TRAIN=1000
NB_TESTS=100

BATCH_SIZE=100
NB_EPOCH=15

SAVE_DIR="models/"

if __name__ == "__main__":
    print("Starting process...")

    if(len(sys.argv) > 2 and sys.argv[1] == "load"):
        print("Loading model...")
        model = load_model(SAVE_DIR+sys.argv[2], 
                {'IO_Heads': IO_Heads})
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
    
    print("Saving model...")
    model.save(SAVE_DIR+save_name)
    
    checkpoint = ModelCheckpoint(SAVE_DIR+save_name)

    print("Training...")
    model.fit(x_in, y_in,
            batch_size=BATCH_SIZE,
            epochs=NB_EPOCH,
            callbacks=[checkpoint]) 

    print("Saving model...")
    model.save(SAVE_DIR+save_name)

    print("Testing model...")
    x_in, y_in = include_batch(NB_TESTS, 
            SEQ_LENGTH, 
            VECTOR_SIZE)  
    
    pred = model.predict(x_in)
    pred = [[p > 0.5 for p in seq] for seq in pred]
    solu = [[p > 0.5 for p in seq] for seq in y_in]
    total = 0

    # print("Pred: ", pred)
    # print("Solu: ", solu)

    for i, p in enumerate(pred):
        ok = sum([1 for x, y in zip(p, solu[i]) if x == y  ])
        total += ok
        # print("Sequence ", i+1, ": ", ok, " / ", SEQ_LENGTH )
    print("Nb sequences: ", NB_TESTS)
    print("Nb tests in each sequence: ", SEQ_LENGTH)
    print("Nb success: ", total, " / ", NB_TESTS*SEQ_LENGTH )

