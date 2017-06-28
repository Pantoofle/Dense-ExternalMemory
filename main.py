import numpy as np
import sys

sys.path.append("layers/")

import keras
from keras.callbacks import *
from keras.layers import * 
from io_heads import *
from data import *

# Params used to generate data 
STATES=5
ALPHABET=5
LENGTH=7

# Network params
MEMORY_SIZE=20
ENTRY_SIZE=5
DEPTH=3
READ_HEADS=3

# Training params
NB_TRAIN=2000
NB_TESTS=100
BATCH_SIZE=1
NB_EPOCH=300

# Dir where models will be saved
SAVE_DIR="models/"

if __name__ == "__main__":
    print("Starting process...")
    print("Getting data...")
    x_in, y_in, tot = automaton_batch(NB_TRAIN, STATES, ALPHABET, LENGTH)
    #  x_in, y_in = include_batch(NB_TRAIN, LENGTH, ALPHABET)

    for i in range(10):
        print("Test ", i, " -> ", x_in[i], "\n  Res: ", y_in[i])
    
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
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
   
    model.summary()
    print("IN: ", x_in.shape)
    print("OUT: ", y_in.shape)


    if not(len(sys.argv) > 1 and sys.argv[1] == "notrain"):
        print("Training ...")
        model.fit(x_in, y_in,
                batch_size=BATCH_SIZE,
                epochs=NB_EPOCH,
                validation_split=0.2,
                callbacks=[
                    #  EarlyStopping(monitor='val_acc', min_delta=0.005, patience = 5),
                    TensorBoard(log_dir='./logs', histogram_freq=1, 
                        write_graph=True, 
                        write_images=False)])

        print("Saving the model...")
        save_model(model, SAVE_DIR+path+".h5")
    
    print("Theoretical: ", 
            VECTOR_SIZE*1./SEQ_LEN * (1.-(1.-1./VECTOR_SIZE)**SEQ_LEN) )

