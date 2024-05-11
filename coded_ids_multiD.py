"""
Class for coded multidimensional IDS trellis
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from helper_functions import *
from tqdm import trange
import time
from coded_ids_multiD_jit import *

import matplotlib
import matplotlib.cm as cm

class coded_ids_multiD():
    def __init__(self, A_in, A_cw, code_trellis_states, code_trellis_edges, code_time_type, \
                 num_traces, p_del, p_sub, p_ins, max_drift = None\
                 , input_prior = None):
        
        self.N_in = (np.array(code_time_type) == "inp").sum()   # input sequence length
        self.N_cw = (np.array(code_time_type) == "out").sum()   # codeword sequence length
        self.code_time_type = code_time_type                    # pointer to indicate whether stage is input or output
        self.A_in = A_in                                        # input alphabet size
        self.A_cw = A_cw                                        # output alphabet size
        
        self.K = num_traces                                     
        self.p_del = p_del
        self.p_sub = p_sub
        self.p_ins = p_ins
        self.max_drift = max_drift
        
        self.p_rep = 1-p_del-p_sub-p_ins                  # assign probability of replication (no del/sub)
        
        if max_drift is None:
            self.max_drift = self.N_cw
            
        if input_prior is None:
            self.input_prior = 1/self.A_in * np.ones((self.N_in,self.A_in))
        else:
            self.input_prior = input_prior
        
        self.check_arg_validity()
        time.sleep(0.4)
        self.make_states(code_trellis_states, code_time_type)
        self.make_edges(code_trellis_states, code_trellis_edges, code_time_type)
        # self.make_edges_dataframe()
        
    def simulate_ids(self, in_seq):
        out = []
        i = 0
        while i < len(in_seq):
            action = np.random.choice(["ins","del","sub","rep"], p = [self.p_ins,self.p_del,self.p_sub,self.p_rep])
            if action == "ins":
                out.append(np.random.choice(self.A_cw))
            elif action == "sub":
                while 1==1:
                    a = np.random.choice(self.A_cw)
                    if a != in_seq[i]:
                        out.append(a)
                        break
                i += 1
            elif action == "rep":
                out.append(in_seq[i])
                i += 1
            elif action == "del":
                i += 1
        
        while 1 == 1:
            action = np.random.choice(["ins","end"], p = [self.p_ins,1-self.p_ins])
            if action == "ins":
                out.append(np.random.choice(self.A_cw))
            else:
                break
        return np.array(out)
                          
    def check_arg_validity(self):
        """
        Check for issues with input args.
        -> Check if the alphabet size is at least 2 since we assume that
        -> check if the ids probabilities are non-negative
        -> check if the ids prob sum exceeds one
        -> check if input prior array is of the right size
        -> check if input priors sum to 1 for each position
        -> check if number of traces is at least 1 and an integer
        """
        if self.A_in < 2 or self.A_cw < 2:
            raise ValueError("IDS module built for alphabets of size at least 2.")
                                                    
        if self.p_del<0 or self.p_sub<0 or self.p_ins<0:
            raise ValueError("IDS probabilities must be non-negative.")
                                                    
        if self.p_rep < 0:
            raise ValueError("Sum of IDS probabilities exceeds 1!")
        
        if self.input_prior.shape != (self.N_in, self.A_in):
            raise ValueError("Input prior array not of appropriate size.")
        
        if not np.allclose(np.ones(self.N_in), self.input_prior.sum(axis = 1)):
            raise ValueError("Input priors should sum to 1 for each row.")
       
        if self.K < 1 or not(isinstance(self.K,int)):
            raise ValueError("Number of traces must be a positive integer.")
        
        # print("Input arguments seem OK, initialization successful.")
        
    def make_states(self, code_states, code_time_type):
        
        N_in = self.N_in
        N_cw = self.N_cw
        A_in = self.A_in
        A_cw = self.A_cw
        K = self.K
        p_del = self.p_del
        p_ins = self.p_ins
        p_sub = self.p_sub
        p_rep = self.p_rep
        max_drift = self.max_drift
        
        n_in = 0
        n_cw = 0
        
        self.states = []
        self.states_idx = []
        self.time_type = []
        
#         for t in trange(len(code_states), desc = "Creating trellis states."):
        for t in range(len(code_states)):

            if code_time_type[t] == "inp" or code_time_type[t] == "end":
                # create channel states and concat to enc states
                min_ptr = max(0, n_cw-max_drift)
                max_ptr = n_cw+max_drift+1
                temp = [np.arange(min_ptr, max_ptr+1)]*K   
                ids_states = cartesian_product(temp)
                joint_states = cartesian_product_mat(ids_states,code_states[t])
                
                self.states.append(joint_states)
                n_in += 1
                
                self.time_type.append(code_time_type[t])
                self.states_idx.append({})
                for idx, state in enumerate(self.states[-1]):
                    self.states_idx[-1][tuple(state)] = idx
            
            elif code_time_type[t] == "out":
                # create channel states and concat with enc states
                # repeat this K+1 times, one for each trace
                min_ptr = max(0, n_cw-max_drift)
                max_ptr = n_cw+max_drift+1
                temp = [np.arange(min_ptr, max_ptr+1)]*K  
                ids_states = cartesian_product(temp)
                
                for k in range(K+1):
                    joint_states = cartesian_product_mat(ids_states,code_states[t]) 
                    
                    self.states.append(joint_states)
                    
                    self.states_idx.append({})
                    for idx, state in enumerate(self.states[-1]):
                        self.states_idx[-1][tuple(state)] = idx
                    
                    if k<K:
                        self.time_type.append("out")
                    else:
                        self.time_type.append("tns")
                        
                n_cw += 1
    
    def make_edges(self, code_states, code_edges, code_time_type):
        
        # reassigning simple names for object attributes
        N_in = self.N_in
        N_cw = self.N_cw
        A_in = self.A_in
        A_cw = self.A_cw
        K = self.K
        p_del = self.p_del
        p_ins = self.p_ins
        p_sub = self.p_sub
        p_rep = self.p_rep
        max_drift = self.max_drift
        
        # create edge attributes as empty lists, we'll append
        # on this list to create the trellis edges
        self.e_from = []
        self.e_to = []
        self.e_w = []
        self.e_type = []
        self.e_w_copy = []
        
        n_in = 0   # input index pointer
        n_cw = 0   # codeword index pointer
        t = 0      # joint encoder-channel stage pointer
        
#         for j in trange(len(code_states), desc = "Creating trellis edges."):
        for j in range(len(code_states)):
            
            if code_time_type[j] == "inp":
                self.e_from.append([])
                self.e_to.append([])
                self.e_type.append([])
                self.e_w.append([])
                
                min_ptr = max(0, n_cw-max_drift)
                max_ptr = n_cw+max_drift+1
                temp = [np.arange(min_ptr, max_ptr+1)]*K   
                ids_states = cartesian_product(temp)
                    
                for ids_state in ids_states:
                    for _, edge in code_edges[j].iterrows():                  
                        from_state = np.concatenate((ids_state, edge.from_state))
                        from_idx = self.states_idx[t][tuple(from_state)]
                        to_state = np.concatenate((ids_state,edge.to_state))
                        to_idx = self.states_idx[t+1][tuple(to_state)]
                        
                        self.e_from[-1].append(from_idx)
                        self.e_to[-1].append(to_idx)
                        self.e_w[-1].append(edge.weight)
                        self.e_type[-1].append("inp")
                
                self.e_from[-1] = np.array(self.e_from[-1])
                self.e_to[-1] = np.array(self.e_to[-1])
                self.e_w[-1] = np.array(self.e_w[-1])
                self.e_type[-1] = np.array(self.e_type[-1])
                self.e_w_copy.append(copy.deepcopy(self.e_w[-1]))
                
                n_in += 1
                t += 1
               
            
            elif code_time_type[j] == "out":
                # create ins/del/sub/rep edges for each trace
                # repeat this K+1 times, one for each trace
                min_ptr = max(0, n_cw-max_drift)
                max_ptr = n_cw+max_drift+1
                temp = [np.arange(min_ptr, max_ptr+1)]*K  
                ids_states = cartesian_product(temp)
                
                for k in range(K):
                    # edges for each trace (ids edges)
                    self.e_from.append([])
                    self.e_to.append([])
                    self.e_type.append([])
                    self.e_w.append([])
                    
                    for ids_state in ids_states:
                        
                        if ids_state.max() < max_ptr:
                            # add insertion edges
                            for code_state in code_states[j]:
                                from_state = np.concatenate((ids_state,code_state))
                                from_idx = self.states_idx[t][tuple(from_state)]
                                to_state = 1*from_state
                                to_state[k] += 1
                                to_idx = self.states_idx[t][tuple(to_state)]
                                
                                self.e_from[-1].append(from_idx)
                                self.e_to[-1].append(to_idx)
                                self.e_w[-1].append(p_ins)
                                self.e_type[-1].append("ins")
                                
                            # add replication and substitution edges
                            for code_state in code_states[j]:
                                from_state = np.concatenate((ids_state,code_state))
                                from_idx = self.states_idx[t][tuple(from_state)]
                                to_state = 1*from_state
                                to_state[k] += 1
                                to_idx = self.states_idx[t+1][tuple(to_state)]
                                
                                # replication edge
                                self.e_from[-1].append(from_idx)
                                self.e_to[-1].append(to_idx)
                                self.e_w[-1].append(p_rep)
                                self.e_type[-1].append("rep")
                                
                                # substitution edge
                                self.e_from[-1].append(from_idx)
                                self.e_to[-1].append(to_idx)
                                self.e_w[-1].append(p_sub)
                                self.e_type[-1].append("sub")
                                
                        # add deletion edges  
                        for code_state in code_states[j]:
                            from_state = np.concatenate((ids_state,code_state))
                            from_idx = self.states_idx[t][tuple(from_state)]
                            to_state = 1*from_state
                            to_idx = self.states_idx[t+1][tuple(to_state)]

                            self.e_from[-1].append(from_idx)
                            self.e_to[-1].append(to_idx)
                            self.e_w[-1].append(p_del)
                            self.e_type[-1].append("del")
                                              
                    self.e_from[-1] = np.array(self.e_from[-1])
                    self.e_to[-1] = np.array(self.e_to[-1])
                    self.e_w[-1] = np.array(self.e_w[-1])
                    self.e_type[-1] = np.array(self.e_type[-1])
                    self.e_w_copy.append(copy.deepcopy(self.e_w[-1]))
                    
                    t += 1
                
                # finally edge to transit to next codeword/input           
                self.e_from.append([])
                self.e_to.append([])
                self.e_type.append([])
                self.e_w.append([])

                min_ptr = max(0, n_cw-max_drift)
                min_ptr_next = max(0,n_cw+1-max_drift)
                max_ptr = n_cw+max_drift+1
                temp = [np.arange(min_ptr, max_ptr+1)]*K  
                ids_states = cartesian_product(temp)                   

                for ids_state in ids_states:
                    if ids_state.min() < min_ptr_next:
                        continue
                    for _, edge in code_edges[j].iterrows():                  
                        from_state = np.concatenate((ids_state, edge.from_state))
                        from_idx = self.states_idx[t][tuple(from_state)]
                        to_state = np.concatenate((ids_state,edge.to_state))
                        to_idx = self.states_idx[t+1][tuple(to_state)]

                        self.e_from[-1].append(from_idx)
                        self.e_to[-1].append(to_idx)
                        self.e_w[-1].append(edge.weight)
                        self.e_type[-1].append("tns")

                self.e_from[-1] = np.array(self.e_from[-1])
                self.e_to[-1] = np.array(self.e_to[-1])
                self.e_w[-1] = np.array(self.e_w[-1])
                self.e_type[-1] = np.array(self.e_type[-1])
                self.e_w_copy.append(copy.deepcopy(self.e_w[-1]))
                    
                t += 1
                n_cw += 1           
            
    def make_edges_dataframe(self):
        
        self.edges = []
        
        for t in trange(len(self.e_from), desc = "Making edge dataframes."):
            self.edges.append([])
            for j in range(len(self.e_from[t])):
                if self.e_type[t][j] == "ins":
                    from_state = self.states[t][self.e_from[t][j]]
                    to_state = self.states[t][self.e_to[t][j]]
                    edge = {"type":self.e_type[t][j],"from_state":from_state,"to_state":to_state,\
                            "weight":self.e_w[t][j],"weight_copy":self.e_w_copy[t][j]}
                    
                    self.edges[-1].append(edge)
                else:
                    from_state = self.states[t][self.e_from[t][j]]
                    to_state = self.states[t+1][self.e_to[t][j]]
                    edge = {"type":self.e_type[t][j],"from_state":from_state,"to_state":to_state,\
                            "weight":self.e_w[t][j],"weight_copy":self.e_w_copy[t][j]}
                    
                    self.edges[-1].append(edge)
            
            self.edges[-1] = pd.DataFrame(self.edges[-1])
            
    def draw_trellis(self, figsize = None, stages = [0,5], fontsize = 15, include_forward_vals = False, \
                            edge_range = [0.0,1.0], state_range = [0.0,1.0], cmap = cm.Greys):
        states = self.states
        edges = self.edges
        state_idx = self.states_idx
        
        if include_forward_vals:
            forward_vals = self.forward_vals
        
        vmin = edge_range[0]
        vmax = edge_range[1]
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        edge_mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

        vmin = state_range[0]
        vmax = state_range[1]
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        state_mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        
        fig = plt.figure(figsize = figsize)
        mid_pt = []
        for t in range(len(states)):
            mid_pt.append(len(states[t])//2)

        for t in trange(stages[0],stages[1], desc = "Drawing trellis."):
            for idx, edge in edges[t].iterrows():
                if edge.type == "ins":
                    if edge.weight_copy >= edge_range[0]:
                        x,y = vertical_arc([t,self.e_from[t][idx]-mid_pt[t]],\
                                        [t,self.e_to[t][idx]-mid_pt[t]])
                        plt.plot(x,y,color = edge_mapper.to_rgba(edge.weight_copy),linewidth = 1)
                    if include_forward_vals:
                        plt.plot(t,self.e_from[t][idx]-mid_pt[t], marker = 'o',color = state_mapper.to_rgba(forward_vals[t][self.e_from[t][idx]]))
                        plt.plot(t,self.e_to[t][idx]-mid_pt[t], marker = 'o',color = state_mapper.to_rgba(forward_vals[t][self.e_to[t][idx]]))
                    else:
                        plt.plot(t,self.e_from[t][idx]-mid_pt[t], marker = 'o',color = 'gray')
                        plt.plot(t,self.e_to[t][idx]-mid_pt[t], marker = 'o',color = 'gray')
                elif edge.type == "sub":
                    if edge.weight_copy >= edge_range[0]:
                        x,y = hanging_line([t,self.e_from[t][idx]-mid_pt[t]],\
                                        [t+1,self.e_to[t][idx]-mid_pt[t+1]])
                        plt.plot(x,y,color = edge_mapper.to_rgba(edge.weight_copy),linewidth = 1)
                elif edge.type == "tns" or edge.type == "inp":
                    if edge.weight_copy >= edge_range[0]:
                        x = [t,t+1]
                        y = [self.e_from[t][idx]-mid_pt[t],self.e_to[t][idx]-mid_pt[t+1]]
                        plt.plot(x,y, color = 'gray',linewidth = 1)
                    if include_forward_vals:
                        plt.plot(t,self.e_from[t][idx]-mid_pt[t], marker = 'o',color = state_mapper.to_rgba(forward_vals[t][self.e_from[t][idx]]))
                        plt.plot(t+1,self.e_to[t][idx]-mid_pt[t+1], marker = 'o',color = state_mapper.to_rgba(forward_vals[t+1][self.e_to[t][idx]]))
                    else:
                        plt.plot(t,self.e_from[t][idx]-mid_pt[t], marker = 'o',color = 'gray')
                        plt.plot(t+1,self.e_to[t][idx]-mid_pt[t+1], marker = 'o',color = 'gray')
                else:
                    if edge.weight_copy >= edge_range[0]:
                        x = [t,t+1]
                        y = [self.e_from[t][idx]-mid_pt[t],self.e_to[t][idx]-mid_pt[t+1]]
                        plt.plot(x,y, color = edge_mapper.to_rgba(edge.weight_copy),linewidth = 1)
                    if include_forward_vals:
                        plt.plot(t,self.e_from[t][idx]-mid_pt[t], marker = 'o',color = state_mapper.to_rgba(forward_vals[t][self.e_from[t][idx]]))
                        plt.plot(t+1,self.e_to[t][idx]-mid_pt[t+1], marker = 'o',color = state_mapper.to_rgba(forward_vals[t+1][self.e_to[t][idx]]))
                    else:
                        plt.plot(t,self.e_from[t][idx]-mid_pt[t], marker = 'o',color = 'gray')
                        plt.plot(t+1,self.e_to[t][idx]-mid_pt[t+1], marker = 'o',color = 'gray')
        
        plt.yticks([])
        plt.xticks(range(stages[1]-stages[0]+1), self.time_type[stages[0]:stages[1]+1], fontsize = fontsize)
        plt.box(False)
        # plt.colorbar(edge_mapper) #Disabling the colorbar to remove errors
        # plt.savefig("temp.png",format = "png", bbox_inches = "tight")
        plt.show()
    
    def draw_optimal_path(self, figsize = None, stages = [0,5], fontsize = 15, \
                            edge_range = [0.0,1.0], state_range = [0.0,1.0], cmap = cm.Greys):
        states = self.states
        edges = self.edges
        state_idx = self.states_idx
        
        forward_vals = self.forward_vals
        
        vmin = edge_range[0]
        vmax = edge_range[1]
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        edge_mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

        vmin = state_range[0]
        vmax = state_range[1]
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        state_mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        
        fig = plt.figure(figsize = figsize)
        mid_pt = []
        for t in range(len(states)):
            mid_pt.append(len(states[t])//2)
        
        stages[1] = min(stages[1], len(edges))
        for t in trange(stages[0],stages[1], desc = "Drawing trellis."):
            for idx, edge in edges[t].iterrows():
                if edge.type == "ins":
                    if edge.weight_copy >= edge_range[0] and forward_vals[t][self.e_from[t][idx]] >= state_range[0] and forward_vals[t][self.e_to[t][idx]] >= state_range[0]:
                        x,y = vertical_arc([t,self.e_from[t][idx]-mid_pt[t]],\
                                        [t,self.e_to[t][idx]-mid_pt[t]])
                        plt.plot(x,y,color = edge_mapper.to_rgba(edge.weight_copy),linewidth = 1)
                        plt.plot(t,self.e_from[t][idx]-mid_pt[t], marker = 'o',color = state_mapper.to_rgba(forward_vals[t][self.e_from[t][idx]]))
                        plt.plot(t,self.e_to[t][idx]-mid_pt[t], marker = 'o',color = state_mapper.to_rgba(forward_vals[t][self.e_to[t][idx]]))
                    
                elif edge.type == "sub":
                    if edge.weight_copy >= edge_range[0] and forward_vals[t][self.e_from[t][idx]] >= state_range[0] and forward_vals[t+1][self.e_to[t][idx]] >= state_range[0]:
                        x,y = hanging_line([t,self.e_from[t][idx]-mid_pt[t]],\
                                        [t+1,self.e_to[t][idx]-mid_pt[t+1]])
                        plt.plot(x,y,color = edge_mapper.to_rgba(edge.weight_copy),linewidth = 1)
                        plt.plot(t,self.e_from[t][idx]-mid_pt[t], marker = 'o',color = state_mapper.to_rgba(forward_vals[t][self.e_from[t][idx]]))
                        plt.plot(t+1,self.e_to[t][idx]-mid_pt[t+1], marker = 'o',color = state_mapper.to_rgba(forward_vals[t+1][self.e_to[t][idx]]))
                        
                elif edge.type == "tns" or edge.type == "inp":
                    if edge.weight_copy >= edge_range[0] and forward_vals[t][self.e_from[t][idx]] >= state_range[0] and forward_vals[t+1][self.e_to[t][idx]] >= state_range[0]:
                        x = [t,t+1]
                        y = [self.e_from[t][idx]-mid_pt[t],self.e_to[t][idx]-mid_pt[t+1]]
                        plt.plot(x,y, color = 'gray',linewidth = 1)
                        plt.plot(t,self.e_from[t][idx]-mid_pt[t], marker = 'o',color = state_mapper.to_rgba(forward_vals[t][self.e_from[t][idx]]))
                        plt.plot(t+1,self.e_to[t][idx]-mid_pt[t+1], marker = 'o',color = state_mapper.to_rgba(forward_vals[t+1][self.e_to[t][idx]]))

                else:
                    if edge.weight_copy >= edge_range[0] and forward_vals[t][self.e_from[t][idx]] >= state_range[0] and forward_vals[t+1][self.e_to[t][idx]] >= state_range[0]:
                        x = [t,t+1]
                        y = [self.e_from[t][idx]-mid_pt[t],self.e_to[t][idx]-mid_pt[t+1]]
                        plt.plot(x,y, color = edge_mapper.to_rgba(edge.weight_copy),linewidth = 1)
                        plt.plot(t,self.e_from[t][idx]-mid_pt[t], marker = 'o',color = state_mapper.to_rgba(forward_vals[t][self.e_from[t][idx]]))
                        plt.plot(t+1,self.e_to[t][idx]-mid_pt[t+1], marker = 'o',color = state_mapper.to_rgba(forward_vals[t+1][self.e_to[t][idx]]))
        
        plt.yticks([])
        plt.xticks(range(stages[1]-stages[0]+1), self.time_type[stages[0]:stages[1]+1], fontsize = fontsize)
        plt.box(False)
        #plt.colorbar(edge_mapper) #Disabling colorbar to remove errors
        plt.show()

    def modify_input_edges(self, input_prior = None):
        """
        Modify the edge weights in the edge weight copy given input prior
        """      
        if input_prior is None:
            pass
        
        else:
            modify_input_edges_jit(self.e_w_copy, self.e_to, self.states, self.time_type, input_prior)

    def compute_e_w(self, tr_list):
        """
        Given the list of output traces, compute the modified edge weights Pr(y|s,s')
        """

        compute_e_w_jit(self.A_cw, self.K, self.e_w_copy, self.e_from,\
                        self.e_type, self.time_type, tr_list, self.states, self.p_ins, self.p_sub, self.p_rep, self.p_del)
    
    def init_bcjr_vals(self, end_ptr, enc_init_state, enc_end_states):
        self.forward_vals, self.backward_vals = init_bcjr_vals_jit(self.states)

        ids_init_state = np.zeros(self.K, dtype = int)
        init_state = np.concatenate((ids_init_state, enc_init_state))
        self.forward_vals[0][self.states_idx[0][tuple(init_state)]] = 1.0
        
        for state in enc_end_states:
            end_state = np.concatenate((end_ptr,state))
            self.backward_vals[-1][self.states_idx[-1][tuple(end_state)]] = 1.0

    def forward_pass(self):
        forward_pass_jit(self.e_from, self.e_to, self.e_w_copy, self.e_type, self.forward_vals)

    def backward_pass(self):
        self.logp_y = backward_pass_jit(self.e_from, self.e_to, self.e_w_copy, self.e_type, self.backward_vals)

    def compute_post(self):
        input_post = np.zeros_like(self.input_prior)
        
        compute_post_jit(self.time_type, self.e_from, self.e_to, self.states, self.e_type,\
             self.e_w_copy, self.forward_vals, self.backward_vals, input_post)
        
        input_post /= input_post.sum(axis = 1)[:,None]
        return input_post 

    def bcjr(self, tr_list, enc_init_state, enc_end_states, input_prior = None):
        
        if len(tr_list) != self.K:
            raise ValueError("Make sure number of traces is {}".format(self.K))

        if input_prior is None:
            input_prior = self.input_prior
        
        self.modify_input_edges(input_prior)
        self.compute_e_w(tr_list)

        end_ptr = np.array([len(tr) for tr in tr_list])
        if end_ptr.max()-self.N_cw > self.max_drift or self.N_cw-end_ptr.min() > self.max_drift:
            raise ValueError("Trace lengths don't fit in allowable drift values.")

        self.init_bcjr_vals(end_ptr, enc_init_state, enc_end_states)

        self.forward_pass()
        self.backward_pass()
        
        ## Compute log(Pr(Y)) for AIR estimation ##
        ids_init_state = np.zeros(self.K, dtype = int)
        init_state = np.concatenate((ids_init_state, enc_init_state)) 
        self.logp_y += np.log2(self.backward_vals[0][self.states_idx[0][tuple(init_state)]])       
        
        return self.compute_post()
        
    
    def init_viterbi_vals(self, end_ptr, enc_init_state, enc_end_state):
        self.forward_vals, self.prev_state, self.prev_state_same_time = init_viterbi_vals_jit(self.states)

        ids_init_state = np.zeros(self.K, dtype = int)
        init_state = np.concatenate((ids_init_state, enc_init_state))
        self.forward_vals[0][self.states_idx[0][tuple(init_state)]] = 1.0

    def viterbi_backtrack(self, end_state):
        current_state = self.states_idx[-1][tuple(end_state)]
        t = len(self.states)-1
        
        out = []
        while t > 0:
            # print(t)
            if self.prev_state_same_time[t][current_state]:
                current_state = self.prev_state[t][current_state]
                continue
            
            else:
                if self.time_type[t-1] == "inp":
                    out.append(self.states[t][current_state][-2])
                current_state = self.prev_state[t][current_state]
                t -= 1

        return np.array(out)[::-1]

    def viterbi(self, tr_list, enc_init_state, enc_end_state, input_prior = None):
        if len(tr_list) != self.K:
            raise ValueError("Make sure number of traces is {}".format(self.K))

        if input_prior is None:
            input_prior = self.input_prior
        
        self.modify_input_edges(input_prior)
        self.compute_e_w(tr_list)

        end_ptr = np.array([len(tr) for tr in tr_list])
        if end_ptr.max()-self.N_cw > self.max_drift or self.N_cw-end_ptr.min() > self.max_drift:
            raise ValueError("Trace lengths don't fit in allowable drift values.")

        self.init_viterbi_vals(end_ptr, enc_init_state, enc_end_state)
        viterbi_pass_jit(self.e_from, self.e_to, self.e_w_copy, self.e_type, \
            self.forward_vals, self.prev_state, self.prev_state_same_time)

        end_state = np.concatenate((end_ptr,enc_end_state))
        return self.viterbi_backtrack(end_state)


######### Additional supporting functions #########


def vertical_arc(point1, point2, res = 10):
    if point1[0] != point2[0]:
        raise ValueError("Points must be vertically aligned.")
    
    pt1 = np.array(point1)
    pt2 = np.array(point2)
    
    dist = 0.5-0.5*np.exp(-np.abs(pt2[1]-pt1[1]))
    temp_pt = (pt1+pt2)*0.5
    temp_pt[0] = pt1[0] - dist
    
    x = []
    y = []
    
    for t in np.linspace(0,1,res):
        B = ((1-t)**2)*pt1 + 2*t*(1-t)*temp_pt + t**2 * pt2
        x.append(B[0])
        y.append(B[1])
    
    return (x,y)

def hanging_line(point1, point2):
    import numpy as np

    a = (point2[1] - point1[1])/(np.cosh(point2[0]) - np.cosh(point1[0]))
    b = point1[1] - a*np.cosh(point1[0])
    x = np.linspace(point1[0], point2[0], 10)
    y = a*np.cosh(x) + b

    return (x,y)