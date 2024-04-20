
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from helper_functions import *
from tqdm import trange
import time

from labellines import labelLine, labelLines # only needed for drawing FSM

class conv_code():
    """
    Build the relevant trellis corresponding to a given convolutional code.
    
    - input is always one symbol (from a fixed arbitrary size alphabet) at one stage
    - outputs are from the quarternary alphabet
    - build one stage of the trellis possibly having multiple outputs
    - modify trellis (by expanding over time and enlarging the state-space) 
      such that each stage correponds to one output
    """
    def __init__(self):
        pass
    
    def bin_cc(self, G):
        """
        Define a CC over binary alphabet and convert it to CC over
        quarternary alphabet, possibly with lower rate.
        - G (array(array({0,1}))): the generator sequences
        """
        K = G.shape[0]
        M = G.shape[1]-1
        
        self.edges = []
        if M == 0:
            self.states = [np.array([0])]
        else:
            self.states = cartesian_product([np.array([0,1])]*M)
        
        for state in self.states:
            for b in [0,1]:
                next_state = np.roll(state,1)
                next_state[0] = b
                bin_out = (np.concatenate((np.array([b]),state)) @ G.T)
                bin_out = np.mod(bin_out,2)
                qrt_out = bin_to_qrt(bin_out)
                
                edge = {"from_state":state, "to_state":next_state\
                        ,"in_symb":b, "out_symb":qrt_out}
                self.edges.append(edge)
                
        self.edges = pd.DataFrame(self.edges)

    def quar_cc(self, G):
        """
        Define a CC over quarternary alphabet and convert it to CC over
        quarternary alphabet, possibly with lower rate.
        - G (array(array({0,1}))): the generator sequences
        """
        K = G.shape[0]
        M = G.shape[1]-1
        
        g = gf4()

        self.edges = []
        if M == 0:
            self.states = [np.array([0])]
        else:
            self.states = cartesian_product([np.array([0,1,2,3])]*M)
        
        for state in self.states:           
            for b in [0,1,2,3]:
                if M == 0:
                    next_state = state
                    qrt_out = g.matmul(np.array([b])[None,:],G.T)[0]
                else:
                    next_state = np.roll(state,1)
                    next_state[0] = b
                    qrt_out = g.matmul(np.concatenate((np.array([b]),state))[None,:],G.T)[0]
                
                edge = {"from_state":state, "to_state":next_state\
                        ,"in_symb":b, "out_symb":qrt_out}
                self.edges.append(edge)
                
        self.edges = pd.DataFrame(self.edges)
                
    def make_trellis(self, in_len, edges = None):
        
        self.in_len = in_len
        
        if edges is None:
            edges = self.edges
        
        enc_states = np.unique(np.array(edges.from_state.tolist()), axis = 0)  # encoder states
        enc_states = enc_states.astype(int)
        in_size = edges.in_symb.max() + 1                                      # input alphabet size
        out_per_in = np.array(edges.out_symb.tolist()).shape[1]                # output bases per input
        self.out_per_in = out_per_in


        # Define "joint states" [enc_state, input, codeword symbol] for each output base per input
        joint_states = []
        for l in range(out_per_in):
            joint_states.append([])
            
            for _,edge in edges.iterrows():
                joint_states[-1].append(list(edge.to_state)+[edge.in_symb]+[edge.out_symb[l]])
                   
            joint_states[-1] = np.array(joint_states[-1], dtype = int)
                
        # Make trellis states over all time steps
        self.trellis_states = []
        self.time_type = []
        
        for n in range(in_len):
            self.trellis_states.append(copy.deepcopy(enc_states))
            self.time_type.append("inp")
             
            for l in range(out_per_in):
                self.trellis_states.append(copy.deepcopy(joint_states[l]))
                self.time_type.append("out")
        
        self.trellis_states.append(copy.deepcopy(enc_states))
        self.time_type.append("end")
        
        # Make trellis edges over all time steps
        self.trellis_edges = []
        
#         for n in trange(in_len, desc = "Making trellis edges"):
        for n in range(in_len):

            temp_edges = []       
            for _,edge in edges.iterrows():
                from_state = edge.from_state
                to_state = list(edge.to_state) + [edge.in_symb] + [edge.out_symb[0]]
                to_state = np.array(to_state, dtype = int)
                weight = 1/in_size
                
                temp_edges.append({"from_state":from_state,"to_state":to_state,"weight":weight})
            
            self.trellis_edges.append(pd.DataFrame(temp_edges))
            
            for l in range(1,out_per_in):
                temp_edges = []
                for _,edge in edges.iterrows():
                    from_state = list(edge.to_state) + [edge.in_symb] + [edge.out_symb[l-1]]
                    from_state = np.array(from_state, dtype = int)
                    to_state = list(edge.to_state) + [edge.in_symb] + [edge.out_symb[l]]
                    to_state = np.array(to_state, dtype = int)
                    weight = 1.0

                    temp_edges.append({"from_state":from_state,"to_state":to_state,"weight":weight})
            
                self.trellis_edges.append(pd.DataFrame(temp_edges))
            
            temp_edges = []
            for _,edge in edges.iterrows():
                from_state = list(edge.to_state) + [edge.in_symb] + [edge.out_symb[-1]]
                from_state = np.array(from_state, dtype = int)
                to_state = edge.to_state
                weight = 1.0

                temp_edges.append({"from_state":from_state,"to_state":to_state,"weight":weight})

            self.trellis_edges.append(pd.DataFrame(temp_edges))
    
    def make_encoder(self):

        self.encoder_lookup = []
#         for t in trange(len(self.trellis_states)-1, desc = "Making encoder lookup table"):
        for t in range(len(self.trellis_states)-1):
            self.encoder_lookup.append({})
            if self.time_type[t] == "inp":
                for __,edge in self.trellis_edges[t].iterrows():
                    # print(edge.to_state)
                    self.encoder_lookup[-1][tuple(edge.from_state),edge.to_state[-2]] = (edge.to_state, edge.to_state[-1])
            elif self.time_type[t+1] == "inp" or self.time_type[t+1] == "end":
                for __,edge in self.trellis_edges[t].iterrows():
                    self.encoder_lookup[-1][tuple(edge.from_state)] = (edge.to_state, None)
            else:
                for __,edge in self.trellis_edges[t].iterrows():
                    self.encoder_lookup[-1][tuple(edge.from_state)] = (edge.to_state, edge.to_state[-1])

    def encode(self, in_seq, init_state = None):
        if len(in_seq) != self.in_len:
            raise ValueError("Ensure input length is {}".format(self.in_len))

        if init_state is None:
            init_state = self.trellis_states[0][0]

        out_seq = []
        n = 0
        current_state = 1*init_state
        for t in range(len(self.trellis_states)-1):
            if self.time_type[t] == "inp":
                current_state, out = self.encoder_lookup[t][tuple(current_state),in_seq[n]]
                out_seq.append(out)
                n += 1
            elif self.time_type[t+1] == "inp" or self.time_type[t+1] == "end":
                current_state, _ = self.encoder_lookup[t][tuple(current_state)]
            else:
                current_state, out = self.encoder_lookup[t][tuple(current_state)]
                out_seq.append(out)

        return np.array(out_seq)

    def puncture(self, redundancy, redundant_positions = None):
        """
        Can only puncture rate 1/2 codes.
        """
        temp_states = []
        temp_edges = []
        temp_time_type = []
        
        if redundant_positions is None:
            redundant_positions = np.random.choice(self.in_len, size = redundancy, replace = False)
            
        self.redundant_positions = redundant_positions

        remove_out = []
        i = 0
        
        for i in range(self.in_len):
            t = i*(self.out_per_in+1)

            if i in redundant_positions:
                remove_out.append(0)
                remove_out.append(0)
                remove_out.append(0)
                continue
            else:
                remove_out.append(0)
                remove_out.append(0)
                remove_out.append(1)
                continue

        for t in range(len(self.time_type)-1):
            if remove_out[t] == 1: 
                connect = {}
                for _,edge in self.trellis_edges[t].iterrows():
                    connect[tuple(edge.from_state)] = edge.to_state
                
                prev_edges = []
                for _,edge in temp_edges[-1].iterrows():
                    temp_edge = {}
                    for key in temp_edges[-1].keys():
                        temp_edge[key] = edge[key]
                    temp_edge["to_state"] = connect[tuple(temp_edge["to_state"])]

                    prev_edges.append(temp_edge)
                
                temp_edges[-1] = None
                temp_edges[-1] = pd.DataFrame(prev_edges)
                

            else:
                temp_time_type.append(self.time_type[t])
                temp_states.append(copy.deepcopy(self.trellis_states[t]))
                temp_edges.append(copy.deepcopy(self.trellis_edges[t]))

        temp_states.append(self.trellis_states[-1])
        temp_time_type.append(self.time_type[-1])

        self.trellis_states = temp_states
        self.trellis_edges = temp_edges
        self.time_type = temp_time_type

    def add_coset(self, coset_vector = None):
        """
        """
        self.trellis_states = copy.deepcopy(self.trellis_states)
        self.trellis_edges = copy.deepcopy(self.trellis_edges)

        out_len = (np.array(self.time_type)=="out").sum()

        if coset_vector is None:
            coset_vector = np.random.choice(4, size = out_len)

        # print(coset_vector)

        j = 0
        for t in range(len(self.time_type)-1):  
            if self.time_type[t] == "out":
                coset_symb = coset_vector[j]
                j += 1  
                self.trellis_states[t][:,-1] += coset_symb 
                self.trellis_states[t][:,-1] = np.mod(self.trellis_states[t][:,-1],4)
            
                keys = self.trellis_edges[t-1].keys()
                edges = []
                for idx, edge in self.trellis_edges[t-1].iterrows():
                    temp = {}
                    for key in keys:
                        # print(key)
                        temp[key] = edge[key]
                    # print(temp)
                    temp["to_state"][-1] += coset_symb
                    temp["to_state"] = np.mod(temp["to_state"],4)
                    edges.append(temp)
                self.trellis_edges[t-1] = None
                self.trellis_edges[t-1] = pd.DataFrame(edges)
                
                edges = []
                for idx, edge in self.trellis_edges[t].iterrows():
                    temp = {}
                    for key in keys:
                        temp[key] = edge[key]
                    temp["from_state"][-1] += coset_symb
                    temp["from_state"] = np.mod(temp["from_state"],4)
                    edges.append(temp)
                self.trellis_edges[t] = None
                self.trellis_edges[t] = pd.DataFrame(edges)
                                



        
    def draw_trellis(self, stages = [0,5], figsize = None):
        states = self.trellis_states
        edges = self.trellis_edges
        
        state_idx = []
        for i in range(len(states)):
            state_idx.append({})
            for idx, state in enumerate(states[i]):
                state_idx[-1][tuple(state)] = idx

        fig = plt.figure(figsize = figsize)
        mid_pt = []
        for t in range(len(states)):
            mid_pt.append(len(states[t])//2)

        for t in range(stages[0],stages[1]):
            for _, edge in edges[t].iterrows():
                x = [t,t+1]
                y = [state_idx[t][tuple(edge.from_state)]-mid_pt[t],state_idx[t+1][tuple(edge.to_state)]-mid_pt[t+1]]
                plt.plot(x,y,marker = 'o', color = 'gray',linewidth = 1)
        
        plt.yticks([])
        plt.xticks(range(stages[0],stages[1]+1),self.time_type[stages[0]:stages[1]+1])
        plt.box(False)
        plt.show()

    def draw_fsm(self, figsize = None):
        edges = self.edges
        states = np.unique(np.array(edges.from_state.tolist()), axis = 0)
        state_idx = {}
        for idx, state in enumerate(states):
            state_idx[tuple(state)] = idx

        fig = plt.figure(figsize = figsize)
        for _, edge in edges.iterrows():
            x = [0,1]
            y = [state_idx[tuple(edge.from_state)],state_idx[tuple(edge.to_state)]]
            plt.plot(x,y,marker = 'o',linewidth = 2, label = str(edge.in_symb)+" / "+str(edge.out_symb))
        
        labelLines(plt.gca().get_lines(),zorder=2.5, fontsize = 12)
        plt.yticks(np.arange(len(states)),states, fontsize = 12)
        plt.xticks([])
        plt.box(False)
        plt.show()


####### Additional functions #########
def bin_to_qrt(s):
    n = len(s)
    out = []
    if n % 2 == 0:
        for i in range(n//2):
            out.append(s[2*i]*2+s[2*i+1])
    else:
        for i in range(n//2):
            out.append(s[2*i]*2+s[2*i+1])
        out.append(s[-1])
    return np.array(out) 