import numpy as np
from numba import njit
from helper_functions import *
from numba.typed import List
np.set_printoptions(precision=2)
@njit
def compute_e_w_jit(A_cw, e_w_copy, e_from, e_type, time_type, trace, states, p_ins, p_sub, p_rep, p_del):
    for t in range(len(e_w_copy)):
        if time_type[t] == "out":
            for idx in range(len(e_w_copy[t])):
                from_state = states[t][e_from[t][idx]]
                ids_state = from_state[0]
                ondeck_cw = from_state[-1]

                if e_type[t][idx] == "ins":
                    if ids_state < len(trace):
                        e_w_copy[t][idx] = p_ins/A_cw
                    else:
                        e_w_copy[t][idx] = 0.0
                elif e_type[t][idx] == "rep":
                    if (ids_state < len(trace)) and (trace[ids_state] == ondeck_cw):
                        e_w_copy[t][idx] = p_rep
                    else:
                        e_w_copy[t][idx] = 0.0
                elif e_type[t][idx] == "sub":
                    if (ids_state < len(trace)) and (trace[ids_state] != ondeck_cw):
                        e_w_copy[t][idx] = p_sub/(A_cw-1)
                    else:
                        e_w_copy[t][idx] = 0.0
                elif e_type[t][idx] == "del":
                    e_w_copy[t][idx] = p_del
    return e_w_copy

@njit
def init_bcjr_vals_jit(states):
    forward_vals = []
    backward_vals = []
    for t in range(len(states)):
        forward_vals.append(0.0 * np.zeros(len(states[t])))
        backward_vals.append(0.0 * np.zeros(len(states[t])))

    return (forward_vals, backward_vals)


@njit
def no_lookahead_pass(e_from, states, e_to, e_type, time_type, e_w_list, forward_val_list, backward_val_list, method = "add", eps = 0.1):

    N_in = 0
    for tt in time_type:
        if tt == "inp":
            N_in += 1

    # Forward pass
    n_in = 0
    out_for = []
    post_probs_for = []
    
    for t in range(len(e_from)):
        if n_in == (N_in//2):
            break
        if time_type[t+1] == "inp":
            n_in += 1
            if method == "add":
                temp_probs = np.zeros(4)
            elif method == "multiply":
                temp_probs = np.ones(4)
            for f in forward_val_list:
                temp_probs_trace = np.zeros(4)
                for idx in range(states[t].shape[0]):
                    state = states[t][idx]
                    a = state[-2]
                    temp_probs_trace[a] += f[t][idx]
                if method == "add":
                    temp_probs += temp_probs_trace
                elif method == "multiply":
                    temp_probs *= temp_probs_trace
            post_probs_for.append(temp_probs.copy()/temp_probs.sum())
               
            best_symb = temp_probs.argmax()
            out_for.append(best_symb)
            for idx in range(states[t].shape[0]):
                state = states[t][idx]
                a = state[-2]
                for f in forward_val_list:
                    if a != best_symb:
                        f[t][idx] *= eps

        for f in forward_val_list:
            f[t] /= f[t].sum()           
        
        for idx in range(len(forward_val_list)):
            f = forward_val_list[idx]
            for i in range(len(e_from[t])):
                from_idx = e_from[t][i]
                to_idx = e_to[t][i]
                if e_type[t][i] == "ins":
                    f[t][to_idx] += (f[t][from_idx] * e_w_list[idx][t][i])
                else:
                    f[t+1][to_idx] += (f[t][from_idx] * e_w_list[idx][t][i])
            f[t] /= f[t].sum()
    
    
    #Backward pass
    out_back = []
    post_probs_back = []

    n_in = 0
    for t in range(len(e_from)-1,0,-1):         
        if n_in == (N_in//2):
            break
        for idx in range(len(backward_val_list)):
            b = backward_val_list[idx]
            for i in range(len(e_from[t])-1,-1,-1):
                from_idx = e_from[t][i]
                to_idx = e_to[t][i]
                if e_type[t][i] == "ins":
                    b[t][from_idx] += (b[t][to_idx] * e_w_list[idx][t][i])
                else:
                    b[t][from_idx] += (b[t+1][to_idx] * e_w_list[idx][t][i])
            b[t] /= b[t].sum()  
            
        if time_type[t-1] == "inp":
            n_in += 1
            if method == "add":
                temp_probs = np.zeros(4)
            elif method == "multiply":
                temp_probs = np.ones(4)
            for b in backward_val_list:
                temp_probs_trace = np.zeros(4)
                for idx in range(states[t].shape[0]):
                    state = states[t][idx]
                    a = state[-2]
                    temp_probs_trace[a] += b[t][idx]
                if method == "add":
                    temp_probs += temp_probs_trace
                elif method == "multiply":
                    temp_probs *= temp_probs_trace
            post_probs_back.append(temp_probs.copy()/temp_probs.sum())    
            best_symb = temp_probs.argmax()
            out_back.append(best_symb)
            for idx in range(states[t].shape[0]):
                state = states[t][idx]
                a = state[-2]
                for b in backward_val_list:
                    if a != best_symb:
                        b[t][idx] *= eps
        
            for b in backward_val_list:
                b[t] /= b[t].sum()
    
    out = out_for + out_back[::-1]
    post_probs = post_probs_for + post_probs_back[::-1]
    return np.array(out), post_probs


@njit
def lookahead_pass(e_from, states, e_to, e_type, time_type, e_w_list, forward_val_list, backward_val_list, method = "add", eps = 0.1):
    
    N_in = 0
    for tt in time_type:
        if tt == "inp":
            N_in += 1
    
    T = len(forward_val_list)
    
    # Forward pass
    out_for = []
    post_probs_for = []
    n_in = 0
    for t in range(len(e_from)):
        if n_in == (N_in//2):
            break
        if time_type[t+1] == "inp":
            n_in += 1
            
            temp_probs = np.ones((T,4))
            
            for i in range(len(forward_val_list)):
                f = forward_val_list[i]
                b = backward_val_list[i]
                temp_probs_trace = np.zeros(4)
                for idx in range(states[t].shape[0]):
                    state = states[t][idx]
                    a = state[-2]
                    temp_probs_trace[a] += (f[t][idx] * b[t][idx])
                temp_probs_trace /= temp_probs_trace.sum()
                temp_probs[i] = temp_probs_trace
                  
            next_priors = np.ones((T,4))
            prod_priors = np.ones(4)
            for p_ in temp_probs:
                prod_priors *= p_
                prod_priors /= prod_priors.sum()
            for idx in range(T):
                for idx2 in range(T):
                    if idx2 != idx:
                        next_priors[idx] *= temp_probs[idx2]
                        
                next_priors[idx] = next_priors[idx]**eps
                next_priors[idx] /= next_priors[idx].sum()
            
            
            post_probs_for.append(prod_priors.copy()/prod_priors.sum())
                    
            best_symb = prod_priors.argmax()
            out_for.append(best_symb)
            
            for idx in range(states[t].shape[0]):
                state = states[t][idx]
                a = state[-2]
                for idx_ in range(len(forward_val_list)):
                    f = forward_val_list[idx_]
                    f[t][idx] *= next_priors[idx_][a]
                    
            for f in forward_val_list:
                f[t] /= f[t].sum()           
        
        for idx in range(len(forward_val_list)):
            f = forward_val_list[idx]
            f[t+1] *= 0.0
            for i in range(len(e_from[t])):
                from_idx = e_from[t][i]
                to_idx = e_to[t][i]
                if e_type[t][i] == "ins":
                    f[t][to_idx] += (f[t][from_idx] * e_w_list[idx][t][i])
                else:
                    f[t+1][to_idx] += (f[t][from_idx] * e_w_list[idx][t][i])
            f[t] /= f[t].sum()
            f[t+1] /= f[t+1].sum()
    
    
    #Backward pass
    out_back = []
    post_probs_back = []
    n_in = 0
    for t in range(len(e_from)-1,0,-1):
        if n_in == (N_in//2):
            break         
        
        for idx in range(len(backward_val_list)):
            b = backward_val_list[idx]
            b[t] *= 0.0
            for i in range(len(e_from[t])-1,-1,-1):
                from_idx = e_from[t][i]
                to_idx = e_to[t][i]
                if e_type[t][i] == "ins":
                    b[t][from_idx] += (b[t][to_idx] * e_w_list[idx][t][i])
                else:
                    b[t][from_idx] += (b[t+1][to_idx] * e_w_list[idx][t][i])
            b[t] /= b[t].sum()  
            
        if time_type[t+1] == "inp" or time_type[t+1] == "end":
            n_in += 1
            
            temp_probs = np.ones((T,4))
            for i in range(len(backward_val_list)):
                b = backward_val_list[i]
                f = forward_val_list[i]
                temp_probs_trace = np.zeros(4)
                for idx in range(states[t].shape[0]):
                    state = states[t][idx]
                    a = state[-2]
                    temp_probs_trace[a] += (f[t][idx] * b[t][idx])
                temp_probs_trace /= temp_probs_trace.sum()
                temp_probs[i] = temp_probs_trace
            
            next_priors = np.ones((T,4))
            prod_priors = np.ones(4)
            for p_ in temp_probs:
                prod_priors *= p_
                prod_priors /= prod_priors.sum()
            for idx in range(T):
                for idx2 in range(T):
                    if idx2 != idx:
                        next_priors[idx] *= temp_probs[idx2]
                next_priors[idx] = next_priors[idx]**eps
                next_priors[idx] /= next_priors[idx].sum()
                            
            post_probs_back.append(prod_priors.copy()/prod_priors.sum())
                    
            best_symb = prod_priors.argmax()
            out_back.append(best_symb)
            
            for idx in range(states[t].shape[0]):
                state = states[t][idx]
                a = state[-2]
                for idx_ in range(len(backward_val_list)):
                    b = backward_val_list[idx_]
                    b[t][idx] *= next_priors[idx_][a]
        
            for b in backward_val_list:
                b[t] /= b[t].sum()
    
    out = out_for + out_back[::-1]
    post_probs = post_probs_for + post_probs_back[::-1]
    return np.array(out), post_probs

# @njit
# def lookahead_pass(e_from, states, e_to, e_type, time_type, e_w_list, forward_val_list, backward_val_list, method = "multiply", eps = 1.0):
    
#     N_in = 0
#     for tt in time_type:
#         if tt == "inp":
#             N_in += 1

#     # Forward pass
#     out_for = []
#     post_probs_for = []
#     n_in = 0
#     for t in range(len(e_from)):
#         if n_in == (N_in//2):
#             break
#         if time_type[t+1] == "inp":
#             n_in += 1
#             if method == "add":
#                 temp_probs = np.zeros(4)
#             elif method == "multiply":
#                 temp_probs = np.ones(4)
#             for i in range(len(forward_val_list)):
#                 f = forward_val_list[i]
#                 b = backward_val_list[i]
#                 temp_probs_trace = np.zeros(4)
#                 for idx in range(states[t].shape[0]):
#                     state = states[t][idx]
#                     a = state[-2]
#                     temp_probs_trace[a] += (f[t][idx] * b[t][idx])
#                 temp_probs_trace /= temp_probs_trace.sum()
#                 if method == "add":
#                     temp_probs += temp_probs_trace
#                 elif method == "multiply":
#                     temp_probs *= temp_probs_trace
#             post_probs_for.append(temp_probs.copy()/temp_probs.sum())
                    
#             best_symb = temp_probs.argmax()
#             out_for.append(best_symb)
#             for idx in range(states[t].shape[0]):
#                 state = states[t][idx]
#                 a = state[-2]
#                 for f in forward_val_list:
#                     if a != best_symb:
#                         f[t][idx] *= eps

#         for f in forward_val_list:
#             f[t] /= f[t].sum()           
        
#         for idx in range(len(forward_val_list)):
#             f = forward_val_list[idx]
#             f[t+1] *= 0.0
#             for i in range(len(e_from[t])):
#                 from_idx = e_from[t][i]
#                 to_idx = e_to[t][i]
#                 if e_type[t][i] == "ins":
#                     f[t][to_idx] += (f[t][from_idx] * e_w_list[idx][t][i])
#                 else:
#                     f[t+1][to_idx] += (f[t][from_idx] * e_w_list[idx][t][i])
#             f[t] /= f[t].sum()
#             f[t+1] /= f[t+1].sum()
    
    
#     #Backward pass
#     out_back = []
#     post_probs_back = []
#     n_in = 0
    
    
#     for t in range(len(e_from)-1,0,-1):
#         if n_in == (N_in//2):
#             break         
        
#         for idx in range(len(backward_val_list)):
#             b = backward_val_list[idx]
#             b[t] *= 0.0
#             for i in range(len(e_from[t])):
#                 from_idx = e_from[t][::-1][i]
#                 to_idx = e_to[t][::-1][i]
#                 if e_type[t][::-1][i] == "ins":
#                     b[t][from_idx] += (b[t][to_idx] * e_w_list[idx][t][::-1][i])
#                 else:
#                     b[t][from_idx] += (b[t+1][to_idx] * e_w_list[idx][t][::-1][i])
#             b[t] /= b[t].sum()  
            
#         if time_type[t+1] == "inp" or time_type[t+1] == "end":
#             n_in += 1
#             if method == "add":
#                 temp_probs = np.zeros(4)
#             elif method == "multiply":
#                 temp_probs = np.ones(4)
#             for i in range(len(backward_val_list)):
#                 b = backward_val_list[i]
#                 f = forward_val_list[i]
#                 temp_probs_trace = np.zeros(4)
#                 for idx in range(states[t].shape[0]):
#                     state = states[t][idx]
#                     a = state[-2]
#                     temp_probs_trace[a] += (f[t][idx] * b[t][idx])
#                 temp_probs_trace /= temp_probs_trace.sum()
#                 if method == "add":
#                     temp_probs += temp_probs_trace
#                 elif method == "multiply":
#                     temp_probs *= temp_probs_trace
#             post_probs_back.append(temp_probs.copy()/temp_probs.sum())
                    
#             best_symb = temp_probs.argmax()
#             out_back.append(best_symb)
#             for idx in range(states[t].shape[0]):
#                 state = states[t][idx]
#                 a = state[-2]
#                 for b in backward_val_list:
#                     if a != best_symb:
#                         b[t][idx] *= eps
        
#             for b in backward_val_list:
#                 b[t] /= b[t].sum()
    
#     out = out_for + out_back[::-1]
#     post_probs = post_probs_for + post_probs_back[::-1]
    
#     return np.array(out), post_probs


def trellis_bma(trellis, traces, enc_init_state, enc_end_states, lookahead = False, method = "multiply", eps = 1.0):
    
    # Reassigning simple names to useful trellis attributes
    p_rep = trellis.p_rep
    p_sub = trellis.p_sub
    p_del = trellis.p_del
    p_ins = trellis.p_ins
    A_cw = trellis.A_cw
    states = trellis.states
    states_idx = trellis.states_idx
    e_from = trellis.e_from
    e_to = trellis.e_to
    e_type = trellis.e_type
    time_type = trellis.time_type
    
    e_w_list = []
    forward_val_list = []
    backward_val_list = []

    for trace in traces:
        if lookahead:
            trellis.bcjr([trace],enc_init_state,enc_end_states)
            forward_vals = trellis.forward_vals
            backward_vals = trellis.backward_vals
            e_w = trellis.e_w_copy
        
        else:
            e_w_copy = copy.deepcopy(trellis.e_w_copy)
            e_w = compute_e_w_jit(A_cw, e_w_copy, e_from, e_type, time_type, trace, states, p_ins, p_sub, p_rep, p_del)
            
            end_ptr = np.array([len(trace)])
            if end_ptr.max()-trellis.N_cw > trellis.max_drift or trellis.N_cw-end_ptr.min() > trellis.max_drift:
                raise ValueError("Trace lengths don't fit in allowable drift values.")
             
            forward_vals, backward_vals = init_bcjr_vals_jit(states)
            
            init_state = np.concatenate(([0],enc_init_state))
            
            forward_vals[0][states_idx[0][tuple(init_state)]] = 1.0
            for enc_end_state in enc_end_states:
                end_state = np.concatenate((end_ptr,enc_end_state))
                backward_vals[-1][states_idx[-1][tuple(end_state)]] = 1.0
        
        e_w_list.append(copy.deepcopy(e_w))
        forward_val_list.append(copy.deepcopy(forward_vals))
        backward_val_list.append(copy.deepcopy(backward_vals))
    
    e_w_typed = List()
    forward_val_typed = List()
    backward_val_typed = List()
    
    for a in e_w_list:
        e_w_typed.append(a)
    for a in forward_val_list:
        forward_val_typed.append(a)       
    for a in backward_val_list:
        backward_val_typed.append(a)
    
    if lookahead:
        out, post_probs = lookahead_pass(e_from, states, e_to, e_type, time_type, e_w_typed,\
                                forward_val_typed, backward_val_typed)
    else:
        out, post_probs = no_lookahead_pass(e_from, states, e_to, e_type, time_type, e_w_typed,\
                                forward_val_typed, backward_val_typed)

    return out, post_probs

def post_multiply(trellis, traces, enc_init_state, enc_end_states, option = "multiply"):
    
    # Reassigning simple names to useful trellis attributes
    p_rep = trellis.p_rep
    p_sub = trellis.p_sub
    p_del = trellis.p_del
    p_ins = trellis.p_ins
    A_cw = trellis.A_cw
    states = trellis.states
    states_idx = trellis.states_idx
    e_from = trellis.e_from
    e_to = trellis.e_to
    e_type = trellis.e_type
    time_type = trellis.time_type
    
    post_probs = None
    
    for trace in traces:
        
        if post_probs is None:
            post_probs = trellis.bcjr([trace],enc_init_state,enc_end_states) * 1.0
        
        else:
            post_probs *= trellis.bcjr([trace],enc_init_state,enc_end_states)

#             post_probs[post_probs < 0.01] = 0.01
            post_probs /= post_probs.sum(axis=1)[:,None]
    
    out = post_probs.argmax(axis=1)

    return out, post_probs
