"""
Auxiliary functions

- integer to base and base to integer conversion
- edge type defined as namedtuples

"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

from numba import njit

def int2base(x, b, size=None):
    """Convert integers to base-b representation
    Args:
        x (list(int)): list (or numpy array) of input values
        b (int): base for conversion
        size (int): integer number of digits to use
    Returns:
        digits (nparray(int)): numpy array of integers
                  digits[i,j] = b**j place digit of i-th input
    Examples:
        >>> print(int2base([10,11],2,size=5)
        [[0, 1, 0, 1, 0],[1, 1, 0, 1 0]]                  
    """
    xx = np.asarray(x)
    if size is None:
        size = int(np.ceil(np.log(np.max(xx))/np.log(b)))
    if isinstance(x,int):
        powers = np.array(b ** np.arange(size))
        digits = (xx // powers) % b
    else:
        powers = np.mat(b ** np.arange(size))
        digits = (xx.reshape(xx.shape + (1,)) // powers) % b
    return digits


def base2int(digits,b):
    """Convert base-b matrix of digits to integer representation
    Args:
        digits (nparray(int)): numpy array (or list)
                  digits[i,j] = b**j place digit of i-th value
        b (int): base for conversion
    Returns:
        x (nparray(int)): numpy array of output values
    Examples:
        >>> print(base2int([[0, 1, 0, 1, 0],[1, 1, 0, 1 0]],2)
        [10, 11]                  
    """
    digits = np.asarray(digits)
    if len(digits.shape)>1:
        powers = np.asarray(b ** np.arange(digits.shape[1]))
    else:
        powers = np.asarray(b ** np.arange(digits.shape[0]))
    return np.matmul(digits,powers)


############## Edge type definitions ############# 

diag_edge = namedtuple("DiagonalEdge",["from_state","to_state","in_label","out_label","out_idx", "weight"])
vert_edge = namedtuple("VerticalEdge",["from_state","to_state","out_label","out_idx","weight"])
"""
diag_edge: an edge across time steps in a trellis
vert_edge: an edge in the same time step in a trellis

- from_state (str or int): the name of the start state for the edge
- to_state (str or int): the name of the end state for the edge
- in_label (str or int): the input label for the edge
- out_label (str or int): the output label for the edge
- out_idx (non-negative int): which output index does the edge correspond to
           for example, in a trellis, each edge at a given stage affects one
           position of the output sequence
- weight (float between 0 and 1): Pr(in_label, out_label, to_state|from_state)

Examples
--------
>> e1 = diag_edge._make([0,1,1,2,0,0.5])
>> print(e1)
>> DiagonalEdge(from_state=0, to_state=1, in_label=1, out_label=2, out_idx=0, weight=0.5)

>> e2 = vert_edge._make([0,1,2,1,0.3])
>> print(e2)
>> VerticalEdge(from_state=0, to_state=1, out_label=2, out_idx=1, weight=0.3)

>> print("Field names of diagonal edge:", diag_edge._fields) # Get field names
>> Field names of diagonal edge: ('from_state', 'to_state', 'in_label', 'out_label', 'out_idx', 'weight')

"""

############# Soft to hard decision decoding ############

def soft2hard(dist):
    """
    Given a symbolwise posterior distribution, pick the most likely symbol 
    at each position.

    Args
    ----
    - dist (list(dict)): where dist[t][s] is the posterior probability of observing
                         symbol 's' at position 't'.

    Returns
    -------
    - out (array): contains most likely symbol at each position
    """
    out = []
    for t in range(len(dist)):
        max_prob = 0.0
        out.append(None)
        for s in dist[t].keys():
            if dist[t][s] >= max_prob:
                max_prob = dist[t][s]
                out[t] = s
    return np.array(out)

#############################################################
################ Cartesian products #########################
#############################################################

########### Cartesian product of arrays ###########

def cartesian_product(arrays):
    """
    Generalized N-dimensional products.
    Given a list of arrays (or array of arrays),
    generates their cartesian product.
    
    E.g: A = [0,1], B = [2,3], C = [4,5]
    
    cartesian_product([A,B,C]) = D
    where D = [
    [0,2,4]
    [0,2,5]
    [0,3,4]
    .
    .
    [1,3,5]
    ]
    
    Parameters
    ----------
    - arrays: list of 1-D arrays or 2-D numpy array
    
    Returns
    -------
    - arr: 2-D numpy array where each row is an element of the
           cartesian product
    """
    la = len(arrays)
    dtype = np.find_common_type([a.dtype for a in arrays], [])
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)

####### Cartesian product of matrices #######

def cartesian_product_mat(A,B):
    """
    Given two matrices A (dimensions N1*N2) and B (M1*M2), returns a
    larger matrix whose rows are the concatenation of every possible 
    pair of rows from A and B.
    """
    out = []
    for a in A:
        for b in B:
            out.append(np.concatenate((a,b)))           
    return np.array(out)


##############################################
def map2int(strarray, chars):
    maps = {}
    for i in range(len(chars)):
        maps[chars[i]] = i
    intarray = np.zeros_like(strarray, dtype = int)
    for i in range(len(strarray)):
        intarray[i] = maps[strarray[i]]
    return intarray

###############################################

###########################################################################
############# Hamming and edit distance, jitted functions #################
###########################################################################

@njit
def hamming_dist(a,b):
    """
    Returns the hamming distance between vectors a and b
    
    Parameters
    ----------
    - a,b: numpy arrays of the same length
    
    Returns
    -------
    - dist: hamming distance between a and b
    
    """
    dist = 0
    for i in range(a.shape[0]):
        if a[i]!=b[i]:
            dist += 1
    
    return dist

@njit
def edit_dist(A, B):
    """
    Given two vectors, compute edit distance (number of insertions, deletions, substitutions
    to transform one vector into the other).
    
    Parameters
    ----------
    - A: numpy array of length N1
    - B: numpy array of length N2
    
    Returns
    -------
    - edit distance between A and B divided by length of A
    
    """
    N1 = A.shape[0]
    N2 = B.shape[0]
    array = np.zeros((N1+1,N2+1))
    for i in range(0,N1):
        array[i+1,0]=i+1
    for j in range(0,N2):
        array[0,j+1]=j+1
    for i in range(0,N1):
        for j in range(0,N2):
            sub_cost = 1
            if A[i] == B[j]:
                sub_cost = 0
            array[i+1,j+1]=min([array[i+1,j]+1,array[i,j+1]+1,array[i,j]+sub_cost])        
    return array[N1,N2]

##############################################################################
### Functions to project each row of a matrix onto the probability simplex ###
##############################################################################

@njit
def simplex_proj(p):
    """
    Function to project a real vector onto the unit simplex hyperplane.
    Algorithm finds closes point on the simplex, i.e., solves the following
    problem
    
    argmin_{x} ||x-p||^2 
    
    s.t. sum(x) = 1
         x_i >= 0 forall i
         
    Check https://eng.ucmerced.edu/people/wwang5/papers/SimplexProj.pdf for
    description of algorithm.
    
    Parameters
    ----------
    - p:  numpy array of length N
    
    Returns
    -------
    - p_proj: numpy array of positive entries such that entries sum to one
    
    """
    A = p.shape[0]
    u = np.sort(p)[::-1]
    
    temp1 = np.zeros(A)
    
    for i in range(A):
        temp1[i] = u[:i+1].sum()
    
    temp2 = (1-temp1) / np.arange(1,A+1)
    
    rho = A
    for i in np.arange(A,0,-1):
        if (u[i-1] + temp2[i-1]) > 0:
            rho = i
            break
    
    lam = temp2[rho-1]
    
    p_proj = np.maximum(np.zeros(A),p+lam)
    return p_proj

@njit
def simplex_proj_mat(P):
    """
    Function to project each row of a matrix P onto the unit simplex
    
    Parameters
    ----------
    - P: N*A numpy array
    
    Returns
    -------
    - P_proj: N*A numpy array of positive entries such that each row sums to one
    """
    P_proj = np.zeros_like(P)
    for i in range(P.shape[0]):
        P_proj[i,:] = simplex_proj(P[i,:])
    
    return P_proj


#######################################################################
###################### Galois field arithmetic ########################
#######################################################################

class gf4():
    """
    -> Creates GF(4) arithmetic tables
    -> vector and matrix multiplication built-in
    """
    def __init__(self):
        self.elements = [0,1,2,3]
        
        # create addition table over GF(4)
        self.add_table = np.array([[0,1,2,3],[1,0,3,2],\
                             [2,3,0,1],[3,2,1,0]])
        
        # Create multiplication table over GF(4)
        self.mul_table = np.array([[0,0,0,0],[0,1,2,3],\
                             [0,2,3,1],[0,3,1,2]])
        
    def add(self,a,b): 
        return self.add_table[a,b]   # add two symbols
    
    def multiply(self,a,b):
        return self.mul_table[a,b]   # multiply two symbols

    def dotprod(self,a,b):           # dot product of 2 vectors
        out = 0
        for i in range(len(a)):
            out = self.add(out,self.multiply(a[i],b[i]))
        return out

    def matmul(self,A,B):            # naive matrix multiplication
        C = np.zeros((A.shape[0],B.shape[1]),dtype = int)

        for i in range(A.shape[0]):
            for j in range(B.shape[1]):
                C[i,j] = self.dotprod(A[i,:],B[:,j])

        return C 
    
###################################################################
################ Achievable information rates #####################
###################################################################

def BCJROR(probs, X):
    probs = np.array(probs)
    N = probs.shape[0]
    I = np.log2(probs[np.arange(N),X]).mean() + 2
    
    return I