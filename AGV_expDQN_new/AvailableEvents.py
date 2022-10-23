import numpy as np

def Enb(state, DFA):
    Events = [];
    M = np.where(DFA == 1)
    N = M[0]   #current state
    O = np.where(N==state)
    Q = M[2]
    for i in O:
        Events.append(Q[i])
    
    return(Events)