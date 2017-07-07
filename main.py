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
from generate_graph import *

# Train params
ALPHABET=2
MIN_LENGTH=2
MAX_LENGTH=6

# Network params
MEMORY_SIZE=4
ENTRY_SIZE=4
DEPTH=1
READ_HEADS=1

# Training params
TRAIN_PER_SIZE=256
BATCH_SIZE=1
NB_EPOCH=10
NB_TRAIN=10
NB_TESTS=10

THRESHOLD=0.5

# Dir where models will be saved
SAVE_DIR="models/"

if __name__ == "__main__":
    print("Starting process...")
    print("Getting automaton...")
    automaton = automaton1()
    #  x_in, y_in = include_batch(NB_TRAIN, LENGTH, ALPHABET)

    #  for i in range(10):
        #  print("Test ", i, " -> ", x_in[i], "\n  Res: ", y_in[i])
    
    #  print("Nb_succes, ", tot, " over ", NB_TRAIN)
    print("Building the first layer...")
    inputs = Input(shape=(None, ALPHABET))

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
    
    model = Model(inputs=inputs, outputs=aux)
    model2 = Model(inputs=inputs, outputs=aux)

    print("Compiling the model...")
    model.compile(optimizer="rmsprop",
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    
    model2.compile(optimizer='adadelta',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
   
    if not(len(sys.argv) > 1 and sys.argv[1] == "notrain"):
        for i in range(NB_TRAIN):        
            length=np.random.randint(MIN_LENGTH, MAX_LENGTH+1)
            if i == 0:
                length = MIN_LENGTH
            print("Train ", i+1, " of ", NB_TRAIN, ": word length: ", length)
            x_in, y_in, tot = automaton_batch(TRAIN_PER_SIZE, ALPHABET, length, automaton=automaton)
            print("Rate: ", tot*1./TRAIN_PER_SIZE, " - ", 1-tot*1./TRAIN_PER_SIZE)
            print("Training ...")
            model.fit(x_in, y_in,
                batch_size=BATCH_SIZE,
                epochs=NB_EPOCH)
            #  model2.fit(x_in, y_in,
            #      batch_size=BATCH_SIZE,
            #      epochs=NB_EPOCH)
    
    print("Generating infered automaton")
    alf_infere(model, automaton, THRESHOLD, ALPHABET)

    #  tpr, fpr = test_network(model, automaton, NB_TESTS, ALPHABET, BATCH_SIZE, NB_WORDS, STOP_RATE, MAX_LENGTH)
    #  trace_ROC(tpr, fpr)

    print("Testing LSTM...")
    #  tpr, fpr = test_network(model2, automaton, NB_TESTS,
    #          STATES, ALPHABET, BATCH_SIZE, NB_WORDS, STOP_RATE, MAX_LENGTH)
    #
