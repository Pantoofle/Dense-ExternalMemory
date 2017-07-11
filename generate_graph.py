import pickle as pi

def automaton1():
    """
    Returns the graph encoding language 1*(01*0)*1*
    """
    return {"entry": {"id":[0]},
            0: {"0": [1],
                "1": [0],
                "exit": [True]},
            1: {"0": [0],
                "1": [1],
                "exit": [False]}
            }

def automaton2():
    """
    Returns the graph encoding language 0A*0 + 1A*1 with A the entire alphabet {0, 1}
    """
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
