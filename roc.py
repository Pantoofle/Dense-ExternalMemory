import plotly.graph_objs as go
import plotly as py
import re
import numpy as np


def trace_ROC(fpr, tpr, path):
    """
    Calls the plotly lib to trace the ROC graph represented by
    tpr the rate of true pos and fpr the rate of false pos
    for different thresholds
    """
    
    plots = []
    trace = go.Scatter(
            x = np.array(fpr[0]),
            y = np.array(tpr[0]),
            name = "Memory network",
            mode = 'markers',
            marker = dict(
                size = 10,
                color = 'rgba(255, 0, 0, .8)'))

    plots += [trace]
    trace = go.Scatter(
            x = np.array(fpr[1]),
            y = np.array(tpr[1]),
            name = "LSTM",
            mode = 'markers',
            marker = dict(
                size = 10,
                color = 'rgba(0, 255, 0, .8)'))


    plots += [trace]
    layout = dict(title = 'ROC',
              yaxis = dict(zeroline = False, range=[-0.1, 1.1]),
              xaxis = dict(zeroline = False, range=[-0.1, 1.1])
             )

    fig = dict(data=plots, layout=layout)    
    py.offline.plot(fig, filename=path)

def mean_roc(path):
    """
    Reads a file containing different roc plots and output a mean roc graph
    when each roc is interpreted as a stair function
    """
    
    # Read the plots
    with open(path, "r") as file:
        x = []
        y = []
        x_i = []
        y_i = []

        txt = file.readlines()
        for l in txt:
            if "=" in l:
                x.append(x_i)
                y.append(y_i)
                x_i = []
                y_i = []
                continue
            
            plot = l.split()
            x_i.append(float(plot[0]))
            y_i.append(float(plot[1]))

    n = len(x)
    
    # Sort entries
    for i in range(n):
        x[i], y[i] =  (list(a) for a in zip(*sorted(zip(x[i], y[i]))))

    times = [sorted(list(set(t))) for t in x]
    values = []
    for i in range(n):
        y_i = []
        for t in times[i]:
            v = [y[i][j] for j in range(len(y[i])) if x[i][j] == t]
            v = sum(v)/len(v)
            y_i.append(v)
        values.append(y_i.copy())

    # Now compute the mean
    ptr = [0]*n
    time = sorted(list(set(sum(times, []))))
    y_res = []
    
    for t in time:
        # Update the pointers
        for i in range(n):
            if ptr[i]+1 == len(times[i]):
                continue
            if x[i][ptr[i]+1] <= t:
                ptr[i] += 1
        
        # Compute the mean
        v = sum([y[i][ptr[i]] for i in range(n)])
        v = (1.*v)/n
        y_res.append(v)

    return time, y_res









