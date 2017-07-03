import numpy as np
import sys

sys.path.append("layers/")

import keras
from keras.callbacks import *
from keras.layers import * 
from io_heads import *
from data import *


# Network params
MEMORY_SIZE=5
ENTRY_SIZE=6
DEPTH=0
READ_HEADS=1

NB_TRAIN_SEQ = 5
TRAIN_SEQ_LEN = 10
PROBA = 0.25

NB_EPOCH = 15
BATCH_SIZE = 1

if __name__ == "__main__":
    print("Starting process...")
    print("Getting data...")
    
    inputs = Input(shape=(TRAIN_SEQ_LEN, 3))

    memory = IO_Heads(units=3,
            vector_size=3,
            memory_size=MEMORY_SIZE, 
            entry_size=ENTRY_SIZE, 
            name="MAIN",
            read_heads=READ_HEADS,
            depth=DEPTH)(inputs)
  
    model = Model(inputs=inputs, outputs=memory)
    print("Compiling the model...")
    model.compile(optimizer='adadelta',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    model.save_weights('models/model.h5')
    model.summary()
    
    
    print("Lets play !")
    
    history = []
    turn = 0
    scoreA = 0
    scoreB = 0
    while True:

        if turn < TRAIN_SEQ_LEN+2 or np.random.random() < PROBA:
            p = np.random.randint(3)
        else:
            model.load_weights('models/model.h5')
            
            training_set = []
            sol = []
            for i in range(NB_TRAIN_SEQ):
                s = np.random.randint(max(turn -TRAIN_SEQ_LEN*2, 0), turn-TRAIN_SEQ_LEN-1)
                training_set += [history[s: s+TRAIN_SEQ_LEN]]
                sol += [history[s+TRAIN_SEQ_LEN]]

            model.fit(training_set, 
                    sol,
                    batch_size=BATCH_SIZE,
                    epochs=NB_EPOCH,
                    verbose=0)
            
            v2 = model.predict(np.array([history[turn-TRAIN_SEQ_LEN :]]), 
                    batch_size = BATCH_SIZE)  
            v2 = v2.tolist()[0]
            p = v2.index(max(v2))
        
        q = int(input("Your move: "))%3
        print("I play: ", p)
        
        if q == p:
            score = str(scoreA) + " - " + str(scoreB)
            print("   Draw " + score)
        elif p == (q+1)%3:
            scoreB += 1
            score = str(scoreA) + " - " + str(scoreB)
            print("\033[91m   I win \033[0m " + score )
        else:
            scoreA += 1
            score = str(scoreA) + " - " + str(scoreB)
            print("\033[92m   You win \033[0m " + score )

        print(" ")
        solution = (q+1)%3
        v = [0., 0., 0.]
        v[solution] = 1.
        history += [v]
        turn += 1
        
















