import numpy as np
import sys

sys.path.append("layers/")

from keras.callbacks import *
import keras.layers
from builder import *
from data import *

VECTOR_SIZE=100
MEMORY_SIZE=1
SEQ_LENGTH=100
DEPTH=0

NB_TRAIN=2000
NB_TESTS=100

BATCH_SIZE=1
NB_EPOCH=25

SAVE_DIR="models/"

if __name__ == "__main__":
    print("Starting process...")
    print("Getting data...")
    x_in, y_in = include_batch(NB_TRAIN, 
            SEQ_LENGTH, 
            VECTOR_SIZE)
    
    print("Building the first layer...")
    inputs = Input(shape=(SEQ_LENGTH, VECTOR_SIZE))
    densed = Dense(MEMORY_SIZE+1, name="Generator")(inputs)
    concat = keras.layers.concatenate([inputs, densed])
    inter = Model(inputs=inputs, outputs=concat)

    #  path = input("Path where to load the first layer: ")
    #  if path == "":
    #      path = "inter"
    #  print("Loading model...")
    #  inter.load_weights(SAVE_DIR+path+".h5")

    #  print("Compiling the first layer...")
    #  inter.compile(optimizer='sgd',
    #                loss='mean_squared_error',
    #                metrics=['accuracy'])
 

    path = input("Path where to load/save the model: ")
    if path == "":
        path = "full"
    #  print("Building the rest of the layer...")
    memory = IO_Heads(memory_size=MEMORY_SIZE, 
        vector_size=VECTOR_SIZE, 
        output_size=VECTOR_SIZE,
        return_sequences=True,
        name="MAIN")(concat)
    
    read = Lambda(lambda x: x[:, :, :VECTOR_SIZE])(memory)
    mem  = Lambda(lambda x: x[:, :, VECTOR_SIZE:])(memory)

    concat2 = keras.layers.concatenate([inputs, read], axis=-1)

    post = Dense(1, activation="sigmoid")(concat2)
    
    model = Model(inputs=inputs, outputs=post)

    if len(sys.argv) > 1 and sys.argv[1]=="load":
        print("Loading the full layer...")
        model = load_model(SAVE_DIR+path+".h5",
                {'IO_Heads': IO_Heads})

    model.save_weights(SAVE_DIR+path+".h5")

    #  model.get_layer("Generator").trainable=False
    
    print("Compiling the model...")
    model.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
   
    model.summary()

    if not(len(sys.argv) > 1 and sys.argv[1] == "notrain"):
        print("Training second layer...")
        model.fit(x_in, y_in,
                batch_size=BATCH_SIZE,
                epochs=NB_EPOCH,
                callbacks=[EarlyStopping(monitor='loss', min_delta=0.01, 
                    patience = 3)])

        print("Saving the full model...")
        save_model(model, SAVE_DIR+path+".h5")

    x_in, y_in = include_batch(NB_TESTS, 
            SEQ_LENGTH,
            VECTOR_SIZE)

    print("Testing Full model...")
    model_pred = model.predict(x_in,
            batch_size=BATCH_SIZE)
    
    #  print("Generating intermediate output")
    #  inter_pred = inter.predict(x_in,
    #          batch_size=BATCH_SIZE)

    print("Generating memory...")
    mem_watcher = Model(inputs=inputs, outputs=mem)
    mem_watcher.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    mem_img = mem_watcher.predict(x_in,
            batch_size=BATCH_SIZE)

    print("Generating last output...")
    comp = Model(inputs=inputs, outputs=concat2)
    comp.compile(optimizer='sgd',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    concaten = comp.predict(x_in,
            batch_size=BATCH_SIZE)

    print("Counting results")
    sol = y_in.flatten()
    pred = model_pred.flatten()
    tot = 0
    for i, x in enumerate(pred):
        if x > 0.5:
            if sol[i] == 1.:
                tot += 1
        else:
            if sol[i] == 0.:
                tot += 1
    print("Score: ", tot, "/", len(sol))

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
