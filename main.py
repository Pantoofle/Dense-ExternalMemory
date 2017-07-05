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
STATES=4
ALPHABET=2
MIN_LENGTH=2
MAX_LENGTH=7

# Network params
MEMORY_SIZE=30
ENTRY_SIZE=10
DEPTH=1
READ_HEADS=2

# Training params
TRAIN_PER_SIZE=300
NB_TRAIN=10

NB_TESTS=1000
BATCH_SIZE=1
NB_EPOCH=10
NB_WORDS=1000
STOP_RATE=0.1

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

    x = []
    y = []   
    print("Generating the predictions")
    for i in range(3, MAX_LENGTH+1):
        x_in, _, _ = automaton_batch(NB_TESTS, STATES, ALPHABET, i, automaton=automaton)
        y_in = model.predict(x_in, batch_size = BATCH_SIZE)
        x += [[str(a.tolist().index(1.)) for a in b] for b in x_in]
        y += y_in.tolist()
        
    x_in = ["".join(w) for w in x]
    L_plus = [x for i, x in enumerate(x_in) if y[i][0] > 0.5]
    L_minus = [x for i, x in enumerate(x_in) if y[i][0] <= 0.5]
    
    print("L_plus: ", len(L_plus))
    print("L_minus: ", len(L_minus))
    print("L_plus: ", L_plus[:10])
    print("L_minus: ", L_minus[:10])

    print("Extracting the infered model from the predictions")
    a2 = extract(L_plus, L_minus)    
    
    a = convert_graph(automaton)

    print("Testing inclusion")
    r = test_inclusion(a, a2, NB_WORDS, stop_rate=STOP_RATE)
    print("A = Original automaton")
    print("B = Infered automaton")
    print("Prop of words valid in A valid in B:     ", r[0])
    print("Prop of words valid in B valid in A:     ", r[1])
    print("Prop of words invalid in A invalid in B: ", r[2])
    print("Prop of words invalid in B invalid in A: ", r[3])

