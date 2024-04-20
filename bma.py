from scipy.stats import mode
import numpy as np

from random import shuffle


def bmala_imp_forward(N, traces, w=2, delay=5, sw=5, mb=4, mf=2):
    K = len(traces)
    p = np.zeros(K, dtype = int)
    a = np.ones(K, dtype = int)
    plu = np.zeros(K, dtype = int)
    istar = np.zeros(K, dtype = int)
    
    estimate = []
    for i in range(N//2):
        plu *= 0
        cur_symbs = [traces[k][p[k]] for k in range(K)]
        cur_symbs = np.array(cur_symbs)
        
        # finding majority vote
        mv = mode(cur_symbs[a==1])[0][0]
        estimate.append(mv)
        
        # looking ahead to find consensus sequence
        la_seqs = []
        for k in range(K):
            if  a[k] == 1 and cur_symbs[k] == mv:
                p[k] += 1
                la_seqs.append(traces[k][p[k]:p[k]+w])
                plu[k] = 1
        
        la_seqs = np.array(la_seqs)
        cons_la = mode(la_seqs, axis = 0)[0][0]
        
        # moving the pointers based on consensus
        for k in range(K):
            if cur_symbs[k] == mv:
                continue
            
            elif a[k] == 1:
                if np.allclose(traces[k][p[k]+1:p[k]+w+1], cons_la):
                    p[k] += 1
                elif np.allclose(traces[k][p[k]+2:p[k]+w+2], cons_la):
                    p[k] += 2
                elif np.allclose(traces[k][p[k]:p[k]+w], cons_la):
                    pass
                else:
                    a[k] = 0
                    istar[k] = i*1
        
        # bringing back inactive traces
        if i >= mb:
            Z = list(estimate[i-mb:i+1]) + list(cons_la[:mf])
            Z = np.array(Z)
            for k in range(K):
                if a[k] == 1:
                    continue

                elif i-istar[k] <= delay:
                    continue

                ptemp = p[k] + (i-istar[k])          
                for j in range(max(ptemp-sw,mb),ptemp+sw+1):
                    if np.allclose(Z,traces[k][j-mb:j+mf+1]):
                        a[k] = 1
                        p[k] = j+1
            
    return np.array(estimate)

def bmala_imp(N, traces, w=2):
    forward = bmala_imp_forward(N, traces, w)
    
    traces_backwards = [trace[::-1] for trace in traces]
    
    backward = bmala_imp_forward(N, traces_backwards, w)[::-1]
    
    return np.array(list(forward)+list(backward))