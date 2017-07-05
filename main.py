import numpy as np
import sys

sys.path.append("layers/")

import keras
from keras.callbacks import *
from keras.layers import * 

from io_heads import *
from data import *
from extractor import *
from graph_compare import *
# Params used to generate data 
STATES=3
ALPHABET=2
MIN_LENGTH=3
MAX_LENGTH=6

# Network params
MEMORY_SIZE=10
ENTRY_SIZE=5
DEPTH=1
READ_HEADS=2

# Training params
TRAIN_PER_SIZE=300
NB_TRAIN=3

NB_TESTS=100
BATCH_SIZE=1
NB_EPOCH=20
NB_WORDS=800
STOP_RATE=0.2

# Dir where models will be saved
SAVE_DIR="models/"

if __name__ == "__main__":
    print("Starting process...")
    print("Getting automaton...")
    automaton = generate_automaton(STATES, ALPHABET) 
    #  x_in, y_in = include_batch(NB_TRAIN, LENGTH, ALPHABET)

    #  for i in range(10):
        #  print("Test ", i, " -> ", x_in[i], "\n  Res: ", y_in[i])
    
    #  print("Nb_succes, ", tot, " over ", NB_TRAIN)
    print("Building the first layer...")
    inputs = Input(shape=(None, ALPHABET))

    path = input("Path where to load/save the model: ")
    if path == "":
        path = "full"

    memory = IO_Heads(units=1,
            vector_size=ALPHABET,
            memory_size=MEMORY_SIZE, 
            entry_size=ENTRY_SIZE, 
            name="MAIN",
            #  return_sequences=True,
            read_heads=READ_HEADS,
            depth=DEPTH)(inputs)
    
    lstm = LSTM(32)(inputs)
    aux = Dense(1)(lstm)
    
    model = Model(inputs=inputs, outputs=memory)
    model2 = Model(inputs=inputs, outputs=aux)

    print("Compiling the model...")
    model.compile(optimizer='adadelta',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    
    model2.compile(optimizer='adadelta',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
   
    if not(len(sys.argv) > 1 and sys.argv[1] == "notrain"):
        for i in range(NB_TRAIN):        
            length=np.random.randint(MIN_LENGTH, MAX_LENGTH+1)
            print("Train ", i+1, " of ", NB_TRAIN, ": word length: ", length)
            x_in, y_in, tot = automaton_batch(TRAIN_PER_SIZE, 
                    STATES, ALPHABET, length, automaton=automaton)
            print("Rate: ", tot*1./TRAIN_PER_SIZE, " - ", 1-tot*1./TRAIN_PER_SIZE)
            print("Training ...")
            model.fit(x_in, y_in,
                batch_size=BATCH_SIZE,
                epochs=NB_EPOCH)
            model2.fit(x_in, y_in,
                batch_size=BATCH_SIZE,
                epochs=NB_EPOCH)
    
    print("Testing Mem...")
    test_network(model, automaton, NB_TESTS, STATES, ALPHABET, BATCH_SIZE, NB_WORDS, STOP_RATE, MAX_LENGTH)
    print("Testing LSTM...")
    test_network(model2, automaton, NB_TESTS, STATES, ALPHABET, BATCH_SIZE, NB_WORDS, STOP_RATE, MAX_LENGTH)

