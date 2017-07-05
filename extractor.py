import numpy as np
import graphviz as gv

def build_tree(L_plus):
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
    fifo = [t["entry"]["id"][0]]
    done = []
    d = 0
    name = {}

    # Computing the new name
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
    # Renaming
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
    if a == m["entry"]["id"][0] or b == m["entry"]["id"][0]:
        m["entry"]["id"][0] = ab
    return m

def reduce(t):

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
    state = t["entry"]["id"][0]
    for c in w:
        if c not in t[state]:
            return False
        state = t[state][c][0]
    return t[state]["exit"][0]

def test_Lminus(t, L):
    for w in L:
        if test_word(t, w):
            return False
    return True


def cmp(x, y):
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
    if len(l) < 2:
        return l
    else:
        a = sort_nodes(l[::2])
        b = sort_nodes(l[1::2])
        c = []
        while len(a) != 0 and len(b) != 0:
            if cmp(a[0], b[0]):
                c +=[a[0]]
                a = a[1:]
            else:
                c +=[b[0]]
                b = b[1:]
        return c+a+b

def generate_order(t):
    nodes = [i for i in t if i != "entry"]
    return sort_nodes(nodes)

def trace_auto(t):
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
    graph.render("img/infered_automaton")

def extract(L_p, L_m):
    t = build_tree(L_p)
    t = rename_depth(t)
    nodes = generate_order(t)
    cont = True
    while cont:
        cont = False
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                t2 = merge(t, nodes[i], nodes[j])
                if len(t2) < len(t):
                    t2 = reduce(t2)
                if not test_Lminus(t2, L_m):
                    t2 = t
                else:
                    t = t2
                    nodes = generate_order(t)
                    cont = True
                    break
            if cont:
                break

    t = rename_depth(t)
    trace_auto(t)
    return t

