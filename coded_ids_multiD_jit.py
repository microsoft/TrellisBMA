import numpy as np
from numba import njit

############ Jitted functions ##########

@njit
def modify_input_edges_jit(e_w_copy, e_to, states, time_type, input_prior_new):
    n = 0
    for t in range(len(e_w_copy)):
        if time_type[t] == "inp":
            for idx, next_state_idx in enumerate(e_to[t]):
                next_state = states[t+1][next_state_idx]
                a = next_state[-2]
                e_w_copy[t][idx] = input_prior_new[n,a]

            n += 1
        

@njit
def compute_e_w_jit(A_cw, K, e_w_copy, e_from, e_type, time_type, tr_list, states, p_ins, p_sub, p_rep, p_del):
    k = 0
    for t in range(len(e_w_copy)):
        if time_type[t] == "out":
            for idx in range(len(e_w_copy[t])):
                from_state = states[t][e_from[t][idx]]
                ids_state = from_state[:K]
                ondeck_cw = from_state[-1]

                if e_type[t][idx] == "ins":
                    if ids_state[k] < len(tr_list[k]):
                        e_w_copy[t][idx] = p_ins/A_cw
                    else:
                        e_w_copy[t][idx] = 0.0
                elif e_type[t][idx] == "rep":
                    if (ids_state[k] < len(tr_list[k])) and (tr_list[k][ids_state[k]] == ondeck_cw):
                        e_w_copy[t][idx] = p_rep
                    else:
                        e_w_copy[t][idx] = 0.0
                elif e_type[t][idx] == "sub":
                    if (ids_state[k] < len(tr_list[k])) and (tr_list[k][ids_state[k]] != ondeck_cw):
                        e_w_copy[t][idx] = p_sub/(A_cw-1)
                    else:
                        e_w_copy[t][idx] = 0.0
                elif e_type[t][idx] == "del":
                    e_w_copy[t][idx] = p_del
            
            k += 1
            if k == K:
                k = 0

@njit
def init_bcjr_vals_jit(states):
    forward_vals = []
    backward_vals = []
    for t in range(len(states)):
        forward_vals.append(0.0 * np.zeros(len(states[t])))
        backward_vals.append(0.0 * np.zeros(len(states[t])))

    return (forward_vals, backward_vals)

@njit
def forward_pass_jit(e_from, e_to, e_w_copy, e_type, forward_vals):
    for t in range(len(e_from)):
        for i in range(len(e_from[t])):
            from_idx = e_from[t][i]
            to_idx = e_to[t][i]
            if e_type[t][i] == "ins":
                forward_vals[t][to_idx] += (forward_vals[t][from_idx] * e_w_copy[t][i])
            else:
                forward_vals[t+1][to_idx] += (forward_vals[t][from_idx] * e_w_copy[t][i])
        forward_vals[t] /= forward_vals[t].sum()

    forward_vals[-1] /= forward_vals[-1].sum()

@njit
def backward_pass_jit(e_from, e_to, e_w_copy, e_type, backward_vals):
    
    logp_y = 0.0    # computing log Pr(Y) from the trellis for AIRs
    logp_y += np.log2(backward_vals[-1].sum())
    backward_vals[-1] /= backward_vals[-1].sum()

    for t in range(len(e_from)-1,-1,-1):
        for i in range(len(e_from[t])):
            from_idx = e_from[t][::-1][i]
            to_idx = e_to[t][::-1][i]
            if e_type[t][::-1][i] == "ins":
                backward_vals[t][from_idx] += (backward_vals[t][to_idx] * e_w_copy[t][::-1][i])
            else:
                backward_vals[t][from_idx] += (backward_vals[t+1][to_idx] * e_w_copy[t][::-1][i])
        logp_y += np.log2(backward_vals[t].sum())
        backward_vals[t] /= backward_vals[t].sum()      
    
    return logp_y

@njit
def compute_post_jit(time_type, e_from, e_to, states, e_type, e_w, F, B, input_post):
    
    n = 0
    for t in range(len(time_type)-1):
        if time_type[t+1] == "inp" or time_type[t+1] == "end":
            for i in range(len(e_from[t])):
                if e_type[t][i] == "ins":
                    continue
                from_idx = e_from[t][i]
                from_state = states[t][from_idx]
                a = from_state[-2]
                to_idx = e_to[t][i]
                input_post[n,a] += (F[t][from_idx] * e_w[t][i] * B[t+1][to_idx])
            
            n += 1     


@njit
def init_viterbi_vals_jit(states):
    forward_vals = []
    prev_state = []
    prev_state_same_time = []
    for t in range(len(states)):
        forward_vals.append(0.0 * np.zeros(len(states[t])))
        prev_state.append(-1*np.ones(len(states[t]),dtype = np.int32))
        prev_state_same_time.append(np.zeros(len(states[t]),dtype = np.int32))

    return (forward_vals, prev_state, prev_state_same_time)


@njit
def viterbi_pass_jit(e_from, e_to, e_w_copy, e_type, forward_vals, prev_state, prev_state_same_time):
    # out = []
    for t in range(len(e_from)):
        for i in range(len(e_from[t])):
            from_idx = e_from[t][i]
            to_idx = e_to[t][i]
            if e_type[t][i] == "ins":
                if forward_vals[t][from_idx]*e_w_copy[t][i] > forward_vals[t][to_idx]:
                    forward_vals[t][to_idx] = forward_vals[t][from_idx]*e_w_copy[t][i]
                    prev_state[t][to_idx] = from_idx
                    prev_state_same_time[t][to_idx] = 1
            else:
                if forward_vals[t][from_idx]*e_w_copy[t][i] > forward_vals[t+1][to_idx]:
                    forward_vals[t+1][to_idx] = forward_vals[t][from_idx]*e_w_copy[t][i]
                    prev_state[t+1][to_idx] = from_idx
        forward_vals[t] /= forward_vals[t].sum()

    forward_vals[-1] /= forward_vals[-1].sum()