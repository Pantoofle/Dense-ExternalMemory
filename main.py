import numpy as np
import sys

sys.path.append("layers/")

import keras
from keras.callbacks import *
from keras.layers import * 
from io_heads import *
from data import *

VECTOR_SIZE=8
SEQ_LEN=7
MEMORY_SIZE=10
ENTRY_SIZE=10

DEPTH=1
READ_HEADS=1

NB_TRAIN=4000
NB_TESTS=100

BATCH_SIZE=1
NB_EPOCH=50

SAVE_DIR="models/"

if __name__ == "__main__":
    print("Starting process...")
    print("Getting data...")
    #  scalar = float(input("Spare time multiplicator: "))
    #  wait, l, x_in, y_in = syracuse_batch(NB_TRAIN, VECTOR_SIZE, scalar)
    x_in, y_in = include_batch(NB_TRAIN, SEQ_LEN, VECTOR_SIZE)

    print("Building the first layer...")
    inputs = Input(shape=(SEQ_LEN, VECTOR_SIZE))

    path = input("Path where to load/save the model: ")
    if path == "":
        path = "full"

    memory = IO_Heads(units=1,
            vector_size=VECTOR_SIZE,
            memory_size=MEMORY_SIZE, 
            entry_size=ENTRY_SIZE, 
            name="MAIN",
            read_heads=READ_HEADS,
            depth=DEPTH,
            return_sequences=True)(inputs)
    
    model = Model(inputs=inputs, outputs=memory)
    mem = model.get_layer("MAIN").memory
    
    if len(sys.argv) > 1 and sys.argv[1]=="load":
        print("Loading the full layer...")
        model = load_model(SAVE_DIR+path+".h5",
                {'IO_Heads': IO_Heads})

    model.save_weights(SAVE_DIR+path+".h5")

    print("Compiling the model...")
    model.compile(optimizer='rmsprop',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
   
    model.summary()
    print("IN: ", x_in.shape)
    print("OUT: ", y_in.shape)
    if not(len(sys.argv) > 1 and sys.argv[1] == "notrain"):
        print("Training second layer...")
        model.fit(x_in, y_in,
                batch_size=BATCH_SIZE,
                epochs=NB_EPOCH,
                validation_split=0.2,
                callbacks=[
                    #  EarlyStopping(monitor='val_acc', min_delta=0.005, patience = 5),
                    TensorBoard(log_dir='./logs', histogram_freq=1, 
                        write_graph=True, 
                        write_images=True)])

    
    print("Theoretical: ", 
            VECTOR_SIZE*1./SEQ_LEN * (1.-(1.-1./VECTOR_SIZE)**SEQ_LEN) )

    #  import matplotlib.pyplot as plt
    #  plt.figure(1)
    #  plt.subplot(245)
    #  plt.imshow(concaten[-1][-4].reshape(1, 2*VECTOR_SIZE))
    #  plt.subplot(246)
    #  plt.imshow(concaten[-1][-3].reshape(1, 2*VECTOR_SIZE))
    #  plt.subplot(247)
    #  plt.imshow(concaten[-1][-2].reshape(1, 2*VECTOR_SIZE))
    #  plt.subplot(248)
    #  plt.imshow(concaten[-1][-1].reshape(1, 2*VECTOR_SIZE))
    #
    #  plt.subplot(241)
    #  plt.imshow(mem_img[-1][-4].reshape((MEMORY_SIZE, VECTOR_SIZE+2)))
    #  plt.subplot(242)
    #  plt.imshow(mem_img[-1][-3].reshape((MEMORY_SIZE, VECTOR_SIZE+2)))
    #  plt.subplot(243)
    #  plt.imshow(mem_img[-1][-2].reshape((MEMORY_SIZE, VECTOR_SIZE+2)))
    #  plt.subplot(244)
    #  plt.imshow(mem_img[-1][-1].reshape((MEMORY_SIZE, VECTOR_SIZE+2)))
    #
    #  plt.savefig("img/memory.png")

    #  plt.clf()
    #  plt.figure(1)
    #  plt.imshow(inter_pred[-1])
    #  plt.savefig("img/inter.png")
