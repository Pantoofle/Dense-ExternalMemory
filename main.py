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
MIN_LENGTH=3
MAX_LENGTH=6

# Network params
MEMORY_SIZE=15
ENTRY_SIZE=10
DEPTH=1
READ_HEADS=2

# Training params
BATCH_SIZE=1
MEM_EPOCH=30
LSTM_EPOCH=50
NB_TRAIN=10
NB_WORDS=5000

# Dir where models will be saved
SAVE_DIR="models/"

if __name__ == "__main__":
    print("Starting process...")
    print("Getting automaton...")
    automaton = automaton1()
    print("Building the models...")
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
    
    model = Model(inputs=inputs, outputs=memory)
    model2 = Model(inputs=inputs, outputs=aux)

    print("Compiling the model...")
    model.compile(optimizer="rmsprop",
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    
    model2.compile(optimizer='rmsprop',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
 
    load = input("Load the models? (y/n) ")
    load = (load == "y")
    train = True

    if load:
        model.load_weights(SAVE_DIR+"DNC.h5")
        model2.load_weights(SAVE_DIR+"LSTM.h5")
        train = (input("Train again? (y/n) ") == "y")    
  
    if train:
        for i in range(NB_TRAIN):        
            length=np.random.randint(MIN_LENGTH, MAX_LENGTH+1)
            if i == 0:
                length = MIN_LENGTH
            print("Train ", i+1, " of ", NB_TRAIN, ": word length: ", length)
            train = 2**(length+1)
            x_in, y_in, tot = automaton_batch(train, ALPHABET, length, automaton=automaton)
            print("Rate: ", tot*1./train, " - ", 1-tot*1./train)
            print("Training DNC ...")
            model.fit(x_in, y_in,
                batch_size=BATCH_SIZE,
                epochs=MEM_EPOCH)
                #  callbacks=[EarlyStopping(monitor='loss', min_delta=0.005, patience=10)])
           
            print("Training LSTM ...")
            model2.fit(x_in, y_in,
                batch_size=BATCH_SIZE,
                epochs=LSTM_EPOCH)
                #  callbacks=[EarlyStopping(monitor='loss', min_delta=0.005, patience=10)])

    save = input("Shall I save the models ? (y/n): ")
    save = (save=="y")
    if save:
        model.save_weights(SAVE_DIR+"DNC.h5")
        model2.save_weights(SAVE_DIR+"LSTM.h5")


    print("Generating infered automaton")
   
    tpr, fpr = test_network(model, automaton, ALPHABET, BATCH_SIZE, NB_WORDS, MIN_LENGTH, MAX_LENGTH)

    tpr2, fpr2 = test_network(model2, automaton, ALPHABET, BATCH_SIZE, NB_WORDS, MIN_LENGTH, MAX_LENGTH)
    
    trace_ROC([tpr, tpr2], [fpr, fpr2])
    
    keep = (input("Keep dnc? ") == "y")
    with open("dnc_dots.txt", "w+") as file:
        
        for t, f in zip(tpr, fpr):
            file.write(str(t)+" "+str(f)+"\n")


    keep = (input("Keep lstm? ") == "y")
    with open("lstm_dots.txt", "w+") as file:
        for t, f in zip(tpr, fpr):
            file.write(str(t)+" "+str(f)+"\n")


