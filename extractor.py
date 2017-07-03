import numpy as np

def build_tree(L_plus):
    n = 1
    tree = {0: {"exit": [False]}}

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

def merge(t, a, b):
    m = {i: {c: [e for e in t[i][c]] for c in t[i]} for i in t}
    # Rewrite each b into a
    for i in m:
        for c in m[i]:
            if c != "exit":
                for j, d in enumerate(m[i][c]):
                    if d == b:
                        m[i][c][j] = a
                        break
                # Delete multiple occurences
                m[i][c] = list(set(m[i][c]))

    # Merge the exiting nodes
    for c in m[b]:
        if c == "exit":
            m[a][c][0] = m[a][c][0] or m[b][c][0]
        else:
            if c in m[a]:
                m[a][c] = list(set(m[b][c] + m[a][c]))
            else:
                m[a][c] = m[b][c]
    m.pop(b) 
    return m

def reduce(t, n):
    m = {i: {c: [e for e in t[i][c]] for c in t[i]} for i in t}
    restart =  True
    while restart:
        for i in m[n]:
            if len(m[n][i]) > 1:
                m = merge(m, m[n][i][0], m[n][i][1])
                break
        restart = False
    return m
