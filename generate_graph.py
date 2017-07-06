import pickle as pi

def automaton1():
    return {"entry": {"id":[0]},
            0: {"0": [1],
                "1": [0],
                "exit": [True]},
            1: {"0": [1],
                "1": [0],
                "exit": [True]}
            }

def automaton2():
    return {"entry": {"id": [0]},
            0: {"0": [1],
                "1": [2],
                "exit": [False]},
            1: {"0": [3],
                "1": [1],
                "exit": [False]},
            2: {"0": [2],
                "1": [4],
                "exit": [False]},
            3: {"0": [3],
                "1": [1],
                "exit": [True]},
            4: {"0": [2],
                "1": [4],
                "exit": [True]}
            }
