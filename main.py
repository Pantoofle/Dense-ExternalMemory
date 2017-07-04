import numpy as np
import sys

sys.path.append("layers/")

import keras
from keras.callbacks import *
from keras.layers import * 
from io_heads import *
from data import *
from extractor import *

# Params used to generate data 
STATES=3
ALPHABET=2
LENGTH=3

# Network params
MEMORY_SIZE=10
ENTRY_SIZE=5
DEPTH=0
READ_HEADS=2

# Training params
NB_TRAIN=500
NB_TESTS=100
BATCH_SIZE=1
NB_EPOCH=50

# Dir where models will be saved
SAVE_DIR="models/"

if __name__ == "__main__":
    print("Starting process...")
    print("Getting data...")
    automaton = generate_automaton(STATES, ALPHABET) 
    x_in, y_in, tot = automaton_batch(NB_TRAIN, STATES, ALPHABET, LENGTH, automaton=automaton)
    #  x_in, y_in = include_batch(NB_TRAIN, LENGTH, ALPHABET)

    print("Rate: ", tot*1./NB_TRAIN)
    #  for i in range(10):
        #  print("Test ", i, " -> ", x_in[i], "\n  Res: ", y_in[i])
    
    #  print("Nb_succes, ", tot, " over ", NB_TRAIN)
    print("Building the first layer...")
    inputs = Input(shape=(LENGTH, ALPHABET))

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
    
    model = Model(inputs=inputs, outputs=memory)

    # Load if asked
    if len(sys.argv) > 1 and sys.argv[1]=="load":
        print("Loading the full layer...")
        model = load_model(SAVE_DIR+path+".h5",
                {'IO_Heads': IO_Heads})

    model.save_weights(SAVE_DIR+path+".h5")

    print("Compiling the model...")
    model.compile(optimizer='adadelta',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
   
    model.summary()
    print("IN: ", x_in.shape)
    print("OUT: ", y_in.shape)


    print("Rate: ", tot*1./NB_TRAIN)
    if not(len(sys.argv) > 1 and sys.argv[1] == "notrain"):
        print("Training ...")
        model.fit(x_in, y_in,
                batch_size=BATCH_SIZE,
                epochs=NB_EPOCH,
                validation_split=0.2)
                #  callbacks=[
                    #  EarlyStopping(monitor='val_acc', min_delta=0.005, patience = 5),
                  #  TensorBoard(log_dir='./logs', histogram_freq=1,
                  #        write_graph=True,
                  #        write_images=False)])
                  #
        print("Saving the model...")
        save_model(model, SAVE_DIR+path+".h5")

    
    x_in, y_in, tot = automaton_batch(NB_TESTS, STATES, ALPHABET, LENGTH, automaton=automaton)
    y = model.predict(x_in, 
            batch_size = BATCH_SIZE)

    x = [[str(a.tolist().index(1.)) for a in x] for x in x_in]

    x_in = ["".join(w) for w in x]
    L_plus = [x for i, x in enumerate(x_in) if y[i][0] > 0.5]
    L_minus = [x for i, x in enumerate(x_in) if y[i][0] <= 0.5]

    extract(L_plus, L_minus)    


