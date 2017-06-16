import numpy as np
import sys

sys.path.append("layers/")

from keras.callbacks import *
from builder import *
from data import *

VECTOR_SIZE=2
MEMORY_SIZE=7
SEQ_LENGTH=2
DEPTH=0
INCLUDE_PROB=0.5

NB_TRAIN=100
NB_TESTS=10

BATCH_SIZE=100
NB_EPOCH=10

SAVE_DIR="models/"

if __name__ == "__main__":
    print("Starting process...")

    if(len(sys.argv) > 3 and sys.argv[1] == "load"):
        print("Loading model...")
        model_mem = load_model(SAVE_DIR+sys.argv[2]+".h5", 
                {'IO_Heads': IO_Heads})
        save_name_mem = sys.argv[2]+".h5"
        model_lstm = load_model(SAVE_DIR+sys.argv[3]+".h5") 
        save_name_lstm = sys.argv[3]+"h5"

    else:
        print("Building new model...")
        model_mem = build_RNN(input_shape=(SEQ_LENGTH, VECTOR_SIZE), 
                memory_size=MEMORY_SIZE, 
                vect_size=VECTOR_SIZE, 
                output_size=1,
                depth=DEPTH)

        model_lstm =  build_LSTM(input_shape=(SEQ_LENGTH, VECTOR_SIZE),
            vect_size=VECTOR_SIZE,
            output_size=1)

        save_name_mem = input("Enter Memory model name: ")
        save_name_lstm = input("Enter LSTM model name: ")
        save_name_mem+= ".h5"
        save_name_lstm+=".h5"

    print("Compiling...")
    model_mem.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    
    model_lstm.compile(optimizer='rmsprop',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    
    model_mem.summary() 
    model_lstm.summary()

    print("Getting data...")
    x_in, y_in = include_batch(NB_TRAIN, 
            SEQ_LENGTH, 
            VECTOR_SIZE)
    
    print("Saving models...")
    model_mem.save(SAVE_DIR+save_name_mem)
    checkpoint_mem = ModelCheckpoint(SAVE_DIR+save_name_mem)
    
    model_lstm.save(SAVE_DIR+save_name_lstm)
    checkpoint_lstm = ModelCheckpoint(SAVE_DIR+save_name_lstm)


    #  print("Training LSTM...")
    #  model_lstm.fit(x_in, y_in,
    #          batch_size=BATCH_SIZE,
    #          epochs=NB_EPOCH,
    #          callbacks=[checkpoint_lstm])

    print("Training Mem...")
    model_mem.fit(x_in, y_in,
            batch_size=BATCH_SIZE,
            epochs=NB_EPOCH,
            callbacks=[checkpoint_mem])

    print("Saving models...")
    model_lstm.save(SAVE_DIR+save_name_lstm)
    model_mem.save(SAVE_DIR+save_name_mem)
    
    print("Testing models...")
    x_in, y_in = include_batch(NB_TESTS, 
            SEQ_LENGTH, 
            VECTOR_SIZE)  
    
    print("LSTM...")
    pred = model_lstm.predict(x_in)


#  print("Questions: ", x_in)
    #  print("Solution: ", y_in)
    #  print("Predictions: ", pred)
    pred = [[p > 0.5 for p in seq] for seq in pred]
    solu = [[p > 0.5 for p in seq] for seq in y_in]
    total = 0
    
    #  print("Pred: ", pred)
    #  print("Solu: ", solu)

    for i, p in enumerate(pred):
        ok = sum([1 for x, y in zip(p, solu[i]) if x == y  ])
        total += ok
        # print("Sequence ", i+1, ": ", ok, " / ", SEQ_LENGTH )
    print("Nb sequences: ", NB_TESTS)
    print("Nb tests in each sequence: ", SEQ_LENGTH)
    print("Nb success: ", total, " / ", NB_TESTS*SEQ_LENGTH )

    print("\nMem...")
    pred = model_mem.predict(x_in)

    print("Saving Last memory state...")
    I= model_mem.layers[0].get_weights()[-1]

    from matplotlib.pyplot import Image
    mem = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(np.uint8)
    img = Image.fromarray(mem)
    img.save('memory.bmp')

    I = x_in[-1]
    xin = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(np.uint8)
    img = Image.fromarray(mem)
    img.save('entry.bmp')



    print("Questions: ", x_in)
    print("Solution: ", y_in)
    print("Predictions: ", pred)
    
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




