###############################################################################
'''
Module for many-body state
11/19/2018
'''
###############################################################################
import numpy as np
import math
from scipy import sparse
from scipy.sparse import linalg
###############################################################################
# Default parameters
base = 2
dtype = 'c16'
SPARSE_MAX = 22
MAT_MAX = 14
###############################################################################
# Operators creation routine
def idn(n,form='coo'):
    '''
    return identity operator in whole Hilbert space
    input:
        n: int
            number of particles
        form: str, optional
            format of sparse matrix, default = COOrdinate format
    output:
        out: [base**n,base**N] (sparse) matrix
            identity operator in whole space
    '''
    assert isinstance (n,int), 'n must be integer!'
    assert n <= SPARSE_MAX, 'n is to large!'
    out = sparse.eye(base**n,dtype=dtype,format=form)
    return out

def opn(op,i,n,form='coo'):
    '''
    return operator in whole Hilbert space from single site operator
    input:
        op: [base,base] matrix
            single site operator
        i: int
            index of the operator
        n: int
            number of particles
        form: str, optional
            format of sparse matrix, default = COOrdinate format
    output: 
        out: [base**n,base**N] (sparse) matrix
            operator in whole space
    '''
    assert isinstance (i,int), 'i must be integer!'
    assert isinstance (n,int), 'n must be integer!'
    assert n <= SPARSE_MAX, 'n is to large!'
    assert 0 <= i <= n, 'i should be in range [0,n]!'
    if i==n:
        out = opn(op,0,n,form = form) 
    else:
        out = idn(i)
        out = sparse.kron(out,op,format=form)
        out = sparse.kron(out,idn(n-i-1),format=form)
    return out

def sx(i,n):
    '''
    return Pauli Sx matrix in whole Hilbert space
    input:
        i: int
            index of the operator
        n: int
            number of particles
    output:
        out: [base**n,base**N] sparse matrix
            identity operator in whole space
    '''
    assert isinstance (i,int), 'i must be integer!'
    assert isinstance (n,int), 'n must be integer!'
    assert n <= SPARSE_MAX, 'n is to large!'
    assert 0 <= i <= n, 'i should be in range [0,n]!'
    s = sparse.coo_matrix(([1,1],([0,1],[1,0])),dtype=dtype)
    out = opn(s,i,n)
    return out
def sy(i,n):
    '''
    return Pauli Sy matrix in whole Hilbert space
    input:
        i: int
            index of the operator
        n: int
            number of particles
    output:
        out: [base**n,base**N] sparse matrix
            identity operator in whole space
    '''
    assert isinstance (i,int), 'i must be integer!'
    assert isinstance (n,int), 'n must be integer!'
    assert n <= SPARSE_MAX, 'n is to large!'
    assert 0 <= i <= n, 'i should be in range [0,n]!'
    if np.dtype(dtype).type in (np.float16,np.float32,np.float64):
        s = sparse.coo_matrix(([-1,1],([0,1],[1,0])),dtype=dtype)     # real Sy matrix
    else:
        s = sparse.coo_matrix(([-1j,1j],([0,1],[1,0])),dtype=dtype)
        out = opn(s,i,n)
    return out
def sz(i,n):
    '''
    return Pauli Sy matrix in whole Hilbert space
    input:
        i: int
            index of the operator
        n: int
            number of particles
    output:
        out: [base**n,base**N] sparse matrix
            identity operator in whole space
    '''
    assert isinstance (i,int), 'i must be integer!'
    assert isinstance (n,int), 'n must be integer!'
    assert n <= SPARSE_MAX, 'n is to large!'
    assert 0 <= i <= n, 'i should be in range [0,n]!'
    s = sparse.coo_matrix(([1,-1],([0,1],[0,1])),dtype=dtype)
    out = opn(s,i,n)
    return out
###############################################################################
# Conversion between product state and Hilbert state
def real2vec(i,n):
    '''
    return product state from index in Hilbert space
    input:
        i: int
            index in whole Hilbeert space
        n: int
            number of particles
    output:
        vec: [n] vector
            product state
    '''
    assert isinstance (i,int), 'i must be integer!'
    assert isinstance (n,int), 'n must be integer!'
    assert 0 <= i < base**n, 'i should be in range [0,base**n)!'
    vec = [0]*n
    rem = i
    for j in range(n):
        vec[j], rem = divmod(rem,2**(n-j-1))
    return vec

def vec2real(v):
    '''
    return index in Hilbert space from product state
    input:
        v: [n] vector
            product state
    output:
        out: int
            index in whole Hilbeert space
    '''
    assert isinstance (v,(list,tuple,np.ndarray)), 'v must be vector!'
    n = len(v)
    out = 0
    for i in range(n):
        out += v[i]*2**(n-i-1)
    out = int(out)
    return out

###############################################################################
# Reduced density matrix & Entropy
def redmatl(v,l):
    '''
    return reduced density matrix of left subsystem
    input:
        v: [n] vector
            pure state in Hilbert space
        l:
            number of particles of left subsystem
    output:
        out: [l,l] matrix
            reduced density matrix
    '''
    assert isinstance (v,(list,tuple,np.ndarray)), 'v must be vector!'
    assert isinstance (l,int), 'n must be integer!'
    assert l <= MAT_MAX, 'l is to large!'
    m = v.reshape([base**l,-1],order='c')
    out = m@m.conj().T
    return out

def redmatr(v,l):
    '''
    return reduced density matrix of right subsystem
    input:
        v: [n] vector
            pure state in Hilbert space
        l:
            number of particles of right subsystem
    output:
        out: [l,l] matrix
            reduced density matrix
    '''
    assert isinstance (v,(list,tuple,np.ndarray)), 'v must be vector!'
    assert isinstance (l,int), 'n must be integer!'
    assert l <= MAT_MAX, 'l is to large!'
    m = v.reshape([-1,base**l],order='f')
    out = m@m.conj().T
    return out

def redmatm(v,start,l):
    '''
    return reduced density matrix of middle subsystem
    input:
        v: [n] vector
            pure state in Hilbert space
        start:
            start index of the subsystem
        l:
            number of particles of middle subsystem
    output:
        out: [l,l] matrix
            reduced density matrix
    '''
    assert isinstance (v,(list,tuple,np.ndarray)), 'v must be vector!'
    assert isinstance (l,int), 'n must be integer!'
    assert l <= MAT_MAX, 'l is to large!'
    m = v.reshape([base**(start+l),-1]).reshape([base**l,-1],order='f')
    out = m@m.conj().T
    return out

def redmato(v,left,right):
    '''
    return reduced density matrix of outer subsystem
    input:
        v: [n] vector
            pure state in Hilbert space
        left:
            number of left particles
        right:
            number of right particles
    output:
        out: [left+right,left+right] matrix
            reduced density matrix
    '''
    assert isinstance (v,(list,tuple,np.ndarray)), 'v must be vector!'
    assert isinstance (left,int), 'left must be integer!'
    assert isinstance (right,int), 'right must be integer!'
    assert left+right <= MAT_MAX, 'left+right is to large!'
    m = v.reshape([base**left,-1],order='f').reshape([base**(left+right),-1])
    out = m@m.conj().T
    return out

def entropy(rho):
    '''
    return entanglement entropy of reduced density matrix
    input:
        rho: [n,n] matrix
            reduced density matrix
    output:
        out: float
            entanglement entropy
    '''
    assert isinstance (rho,np.ndarray), 'v must be matrix!'
    r = np.linalg.eigvalsh(rho)
    r = r[r>0]
    s = -r@np.log(r)
    out = max(0,s)
    return out
###############################################################################
# Messurement
def num(v):
    '''
    return number of particles from a vector in Hilbert space
    input:
        v: [n] vector
            reduced density matrix
    output:
        n: int
            number of particles
    '''
    assert isinstance (v,(list,tuple,np.ndarray)), 'v must be vector!'
    n = int(math.log(len(v),base))
    return n

def mesure(v):
    '''
    return expectation of each quantum state of each particle
    input:
        v: [...,n] vector/matrix
            vector (stack of vectors) of Hilbert space
    output:
        out: [...,n,l] array
            expectation of each quantum state of each particle
    '''
    assert isinstance (v,np.ndarray), 'v must be numpy array!'
    if v.dim == 1:
        l = int(math.log(len(v),base))
        out = np.empty([l,base])
        m = np.abs(v)**2
        for i in range(l-1):
            m = np.reshape(m,[base,-1])
            out[i] = m.sum(1)
            m = m.sum(0)
        out[l-1] = m
    else:
        assert v.dim == 2, 'v must be of dimension 1 or 2!'
        n,l = v.shape
        l = int(math.log(l,base))
        out = np.empty([n,l,base])
        for i in range(n):
            out[i] = mesure(v[i])
    return out

def spin(v):
    '''
    return expectation of spin of each particle
    input:
        v: [...,n] vector/matrix
            vector (stack of vectors) of Hilbert space
    output:
        out: [...,n,3] array
            expectation of spin of each particle
    '''
    global base
    assert base == 2, 'spin can only be calculate for base 2'
    assert isinstance (v,np.ndarray), 'v must be numpy array!'
    def spin_mul(a,b): # inner function to cauculate muti spin
        z = np.conj(a)@a-np.conj(b)@b
        b = 2*np.conj(a)@b
        x = b.real
        y = b.imag
        return [x,y,z.real]
    if v.dim == 1:
        l = int(math.log2(len(v)))
        out = np.empty([l,3])
        for i in range(l):
            temp = v.reshape([2**(i+1),-1]).reshape([2,-1],order='f')
            out[i] = spin_mul(*temp)
    else:
        assert v.dim == 2, 'v must be of dimension 1 or 2!'
        n,l = v.shape
        l = int(math.log2(l))
        out = np.empty([n,l,3])
        for i in range(n):
            out[i] = spin(v[i])
    return out 

def spinz(v):
    '''
    return expectation of spin z of each particle
    input:
        v: [...,n] vector/matrix
            vector (stack of vectors) of Hilbert space
    output:
        out: [...,n] array
            expectation of spin z of each particle
    '''
    temp = mesure(v)
    out = -np.diff(temp)
    return out
###############################################################################
# State evolement routine
def evo(H,v,t):
    '''
    return the vector in Hilbert space after evolving time t
    input:
        H: [n,n] (sparse) matrix
            Hamiltonion
        v: [n] vector/matrix
            vector of Hilbert space
        t: float
            evolving time
    output:
        out: [n] vector
            vector after evolving time t
    '''
    assert isinstance (H,(np.ndarray,sparse.spmatrix)), 'v must be numpy matrix or sparse matrix!'
    assert isinstance (v,np.ndarray), 'v must be vector!'
    assert H.shape == (len(v),)*2, 'H must be of the same length as v!'
    assert t.imag == 0, 't must be real!'
    out = linalg.expm_multiply(-1j*H*t,v)
    return out
def evom(H,v,start,end,num):
    '''
    return stack of vectors in Hilbert space after multiple evolving times
    input:
        H: [n,n] (sparse) matrix
            Hamiltonion
        v: [n] vector/matrix
            vector of Hilbert space
        start: float
            start time
        end: float
            end time
        num: int
            number of division
    output:
        out: [num,n] matrix
            stack of vector after multiple evolving times
    '''
    assert isinstance (H,(np.ndarray,sparse.spmatrix)), 'v must be numpy matrix or sparse matrix!'
    assert isinstance (v,np.ndarray), 'v must be vector!'
    assert H.shape == (len(v),)*2, 'H must be of the same length as v!'
    assert start.imag == 0 , 'start must be real!'
    assert end.imag == 0 , 'end must be real!'
    out = linalg.expm_multiply(-1j*H,v,start=start,stop=end,num=num)
    return out

if __name__ == '__main__':
    pass