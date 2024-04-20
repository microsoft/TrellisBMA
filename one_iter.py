from helper_functions import *
import pandas as pd
import numpy as np
from scipy.stats import mode

from Levenshtein import distance, editops
from scipy.stats import mode
from tqdm import trange

from conv_code import *
from coded_ids_multiD import *
from bma import *
from trellis_bma import *

import time
import multiprocessing as mp


def one_iter_rest(in_len, in_seq, tr_list, cluster_size, N_cw, ids_trellis, cc):
    
    max_drift = ids_trellis.max_drift
    
    traces = []
    for tr in tr_list:
        if np.abs(len(tr)-N_cw) <= max_drift:
            traces.append(tr)
        elif len(tr) > N_cw:
            idx_delete = np.random.choice(len(tr),len(tr)-N_cw-max_drift, replace = False)
            traces.append(np.delete(tr,idx_delete))
        else:
            idx_insert = np.random.choice(len(tr),N_cw-max_drift-len(tr), replace = False)
            traces.append(np.insert(tr,idx_insert,0))

    bma_estimate = bmala_imp(N_cw,tr_list,2)

    post_mul_estimate, post_mul_probs = post_multiply(ids_trellis,traces,cc.trellis_states[0][0],\
                                             cc.trellis_states[-1])

    Tbma_noLA_estimate, Tbma_noLA_probs = trellis_bma(ids_trellis,traces,cc.trellis_states[0][0],\
                                             cc.trellis_states[-1],lookahead = False)
    Tbma_LA_estimate, Tbma_LA_probs = trellis_bma(ids_trellis,traces,cc.trellis_states[0][0],\
                                             cc.trellis_states[-1],lookahead = True)

    
    bma_error_ = (in_seq != bma_estimate).sum()
    post_mul_error_ = (in_seq != post_mul_estimate).sum()
    Tbma_noLA_error_ = (in_seq != Tbma_noLA_estimate).sum()
    Tbma_LA_error_ = (in_seq != Tbma_LA_estimate).sum()
    
    post_mul_BOR_ = BCJROR(post_mul_probs,in_seq)
    Tbma_noLA_BOR_ = BCJROR(Tbma_noLA_probs, in_seq)
    Tbma_LA_BOR_ = BCJROR(Tbma_LA_probs, in_seq)
    
    return (bma_error_, post_mul_error_, Tbma_noLA_error_, Tbma_LA_error_,\
           post_mul_BOR_,Tbma_noLA_BOR_,Tbma_LA_BOR_)