import numpy as np
import graphviz as gv

# Where the alf executable and lib can be found
ALF_PATH="libalf/online"
ALF_ENV ={"LD_LIBRARY_PATH":"/usr/local/lib"}

def rand_walk(t, stop_rate=0.25, length=0):
    """
    Generate a word over t
    It may or may not end in a final state
    After each step, the walk has probability stop_rate to just end there
    If length is specified, then the length is respected
    """
    
    word = []
    state = t["entry"]["id"][0]
    l = 0

    while True:
        possib = [c for c in t[state] if c != "exit"]
        if len(possib) == 0:
            #  print("No place to go...")
            break
        i = np.random.randint(len(possib))
        c = possib[i]
        #  print("Next move: ", c)

        word += [c]
        l += 1

        state = t[state][c][0]
        
        if (stop_rate > np.random.random() and length == 0) or l == length:
            #  print("The random said... STOP")
            break

    return t[state]["exit"][0], word



def build_tree(L_plus):
    """
    Builds the initial tree for the RPNI method to infer automaton, from L_plus,
    the list of accepted words
    """
    
    n = 1
    tree = {0: {"exit": [False]}, 
            "entry": {"id": [0]}}

    for s in L_plus:    
        actual = 0
        for c in s:
            if c not in tree[actual]:
                tree[actual][c] = [n]
                tree[n] = {"exit": [False]}
                n += 1
            actual = tree[actual][c][0]
        tree[actual]["exit"] = [True]

    return tree

def rename_depth(t):
    """
    Rename the whole graph in depth-first order
    """
    fifo = [t["entry"]["id"][0]]
    done = []
    d = 0
    name = {}

    # Computing the new name of each node
    while fifo != []:
        e = fifo[0]
        fifo = fifo[1:]
        for c in t[e]:
            if c == "exit":
                continue
            for i in t[e][c]:
                if i not in done:
                    fifo += [i]
        
        if e not in done:
            name[e] = d
            d += 1
            done += [e]

    # Renaming nodes and transitions
    m = {}
    for e in t:
        m[e] = {}
        for c in t[e]:
            if c == "exit":
                m[e][c] = [t[e][c][0]]
            else:
                m[e][c] = []
                for d in t[e][c]:
                    m[e][c].append(str(name[d]))

        if e == "entry":
            m[e] = {"id": [str(name[t[e]["id"][0]])]}
        else :
            tmp = m.pop(e)
            m[str(name[e])] = tmp
    return m

        

def merge(t, a, b):
    """
    Merging nodes a and b from graph t
    The resulting graph may be undeterministic
    """

    m = {i: {c: [e for e in t[i][c]] for c in t[i]} for i in t}
    # Rewrite each b and a into ab
    ab = a + "." + b
    for i in m:
        if i == "entry":
            continue
        for c in m[i]:
            if c != "exit":
                for j, d in enumerate(m[i][c]):
                    if d == b or d == a:
                        m[i][c][j] = ab
                # Delete multiple occurences
                m[i][c] = list(set(m[i][c]))
    
    # Merge the exiting nodes
    if b not in m or a not in m:
        return m

    for c in m[b]:
        if c == "exit":
            m[a][c][0] = m[a][c][0] or m[b][c][0]
        else:
            if c in m[a]:
                m[a][c] = list(set(m[b][c] + m[a][c]))
            else:
                m[a][c] = m[b][c]
    m[ab] = m[a]
    m.pop(a)
    m.pop(b) 
    # Special case of entry node merged
    if a == m["entry"]["id"][0] or b == m["entry"]["id"][0]:
        m["entry"]["id"][0] = ab
    return m

def reduce(t):
    """
    Merges conflict nodes until t becomes deterministic
    """

    m = {i: {c: [e for e in t[i][c]] for c in t[i]} for i in t}
    restart =  True
    while restart:
        restart = False
        for n in m:
            for i in m[n]:
                if len(m[n][i]) > 1:
                    m = merge(m, m[n][i][0], m[n][i][1])
                    restart = True
                    break
            if restart:
                break
    return m

def test_word(t, w):
    """
    Tests wether w is accepted by automaton t or not
    """

    state = t["entry"]["id"][0]
    for c in w:
        if c not in t[state]:
            return False
        state = t[state][c][0]
    return t[state]["exit"][0]

def test_Lminus(t, L):
    """
    Tests if a word in L is accepted by automaton t
    """

    for w in L:
        if test_word(t, w):
            return False
    return True


def cmp(x, y):
    """
    Lexicographic-like order on nodes names
    """

    a = x.split(".")
    b = y.split(".")

    if len(a) < len(b):
        return True
    elif len(a) > len(b):
        return False
    else:
        i = 0
        while int(a[i]) == int(b[i]):
            i+=1

        return int(a[i]) < int(b[i])

def sort_nodes(l):
    """
    Merge sorts the list of nodes l with previous order
    """
    
    if len(l) < 2:
        return l
    else:
        a = sort_nodes(l[::2])
        b = sort_nodes(l[1::2])
        c = []
        i, j = 0, 0
        while i != len(a) and j != len(b):
            if cmp(a[i], b[j]):
                c +=[a[i]]
                i += 1
            else:
                c +=[b[j]]
                j += 1
        if i == len(a):
            return c+b[j:]
        else:
            return c+a[i:]

def generate_order(t):
    """
    extracts the list of nodes of t and returns them sorted
    """

    nodes = [i for i in t if i != "entry"]
    return sort_nodes(nodes)

def trace_auto(t, path):
    """
    Trace given automaton
    """
    
    graph = gv.Digraph(format="svg")
    for i in t:
        if i == "entry":
            graph.node(i)
            graph.edge(i, t[i]["id"][0])
            continue
        graph.node(i)
        for c in t[i]:
            if c == "exit":
                if t[i][c][0]:
                    graph.edge(i, str(c))
                continue
            for j in t[i][c]:
                graph.edge(i, j, str(c))
    graph.render(path)

def extract(L_p, L_m):
    """
    Applies RPNI method to infer an automaton accepting L_p 
    and refusing L_m
    """

    print("Extracting...")
    t = build_tree(L_p)
    t = rename_depth(t)
    nodes = generate_order(t)
    cont = True
    while cont:
        cont = False
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                print("Merging ", nodes[i]," - ", nodes[j])
                t2 = merge(t, nodes[i], nodes[j])
                if len(t2) < len(t):
                    print("Reducing...")
                    t2 = reduce(t2)
                if not test_Lminus(t2, L_m):
                    t2 = t
                    print("A bad one")
                else:
                    t = t2
                    nodes = generate_order(t)
                    cont = True
                    print("A good one")
                    break
            if cont:
                break

    t = rename_depth(t)
    return t


def word2vect(w, alphabet):
    """
    Rewrites word w in "one-hot" format
    """
    
    x = np.zeros((1, len(w), alphabet))
    for j in range(len(w)):
        x[0][j][int(w[j])] = 1.
    return x



def alf_infere(model, automaton, threshold, alphabet):
    """
    Calls libalf algorithmes to infere the automaton learned by model
    automaton is the solution and is used to do random walks when the 
    algo needs a word
    """

    # Calling the alf algo
    import subprocess as sp
    s = sp.Popen([ALF_PATH], 
            stdout=sp.PIPE, 
            stdin=sp.PIPE, 
            universal_newlines=True,
            env=ALF_ENV)
    i = s.stdin
    o = s.stdout
    
    conj = ""

    # Sending the alphabet size to alf
    o.readline()
    i.write(str(alphabet) + "\n")
    i.flush()

    # Because the network can't work with empty words, we answer manually
    o.readline()
    i.write(str(1) + "\n")
    i.flush()

    read = True

    while True:
        if read:
            w = o.readline()[:-1]
        else:
            read = True
        #  print("He said: ",w)
    
        # When alf makes a conjecture
        if w == "Conjecture:":
            #  print("Oh, a try! Bring it on!")
            c = ""
            # Read the conjecture
            while w != "End of auto":
                w = o.readline()[:-1]
                c += w

            # If it is the same as the last time, we reached the automaton
            if c == conj:
                #  print("Okay!")
                i.write("y\n")
                i.flush()
                continue
            else:
                #  print("Nope!")
                conj = c
                i.write("n\n")
                i.flush()
                
                # Else we give a random word to continue the algo
                w = o.readline()[:-1]
                _, word = rand_walk(automaton)
                word = "".join(word)
                i.write(word + "\n")
                i.flush()
                
                w = o.readline()[:-1]

                # We may be unlucky and have given a word already given. so we stop there
                if w == "Conjecture:":
                    i.write("y\n")
                    i.flush()
                    read = False
                    continue

                # Else, we predict the word we just gave
                y = model.predict(word2vect(word, alphabet), batch_size=1)
                r = str(int(y[0][0] > threshold)) + "\n"
                i.write(r)
                i.flush()
                continue  
        
        if w == "Result:":
            # The algo stopped and gave an answer
            res = o.read()
            #  print("Result:\n", res)
            return res
        
        # Else, the common scenario, we translate the word requested and answer the question
        x= word2vect(w, alphabet)

        y = model.predict(x, batch_size=1)
        r = str(int(y[0][0] > threshold)) + "\n"
        i.write(r)
        i.flush()
        #  print("I answered ",r)



