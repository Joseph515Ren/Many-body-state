###############################################################################
'''
Module for many-body state
11/30/2018
'''
###############################################################################
import numpy as np
import math
from scipy import sparse
from scipy.sparse import linalg
import sys
###############################################################################
# Default parameters
base = 2
dtype = 'c16'
SPARSE_MAX = 22
MAT_MAX = 14
###############################################################################
# Operators creation routine
#------------------------------------------------------------------------------
def empty(n):
    '''
    Empty matrix for n particle Hilbert space
    
    Return sparse empty matrix.
    
    Parameters
    ----------
    n : int
        Number of particles
        
    Returns
    -------
    out : [base**n,base**n] (sparse) matrix
        Empty operator in whole space
    '''
    assert isinstance (n,int), 'n must be integer!'
    assert n <= SPARSE_MAX, 'n is to large!'
    out = sparse.coo_matrix((2**n,2**n),dtype=dtype)
    return out
#------------------------------------------------------------------------------
def idn(n,form='coo'):
    '''
    Identity matrix for n particle Hilbert space
    
    Return sparse diagnal matrix.
    
    Parameters
    ----------
    n : int
        Number of particles
    form : str, optional
        Sparse format of the result, default = COOrdinate format
        
    Returns
    -------
    out : [base**n,base**n] (sparse) matrix
        Identity operator in whole space
    '''
    assert isinstance (n,int), 'n must be integer!'
    assert n <= SPARSE_MAX, 'n is to large!'
    out = sparse.eye(base**n,dtype=dtype,format=form)
    return out
#------------------------------------------------------------------------------
def opn(op,i,n,form='coo'):
    '''
    Single-body operator in many-body space
    
    Return kronecker product of single-body operator on site i and identity
    operator on other sites.
    
    Parameters
    ----------
    op : [base,base] matrix
        Single site operator
    i : int
        Index of the operator
    n : int
        Number of particles
    form : str, optional
        Format of sparse matrix, default = COOrdinate format
    
    Returns
    ------- 
    out: [base**n,base**N] (sparse) matrix
        Operator in whole space
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
###############################################################################
# Conversion between product state and Hilbert state
#------------------------------------------------------------------------------
def real2vec(i,n):
    '''
    Convertion from index of many-body state to real occupation state
    
    Return array representing the occupation of each site.
    
    Parameters
    ----------
    i : int
        Index in whole Hilbeert space
    n : int
        Number of particles
        
    Returns
    -------
    vec: [n] vector
        Occupation
    
    Examples
    --------
    >>> real2vec(0,5)
    >>> [0, 0, 0, 0, 0]

    >>> real2vec(1,5)
    >>> [0, 0, 0, 0, 1]

    >>> real2vec(3,5)
    >>> [0, 0, 0, 1, 1]
    '''
    assert isinstance (i,int), 'i must be integer!'
    assert isinstance (n,int), 'n must be integer!'
    assert 0 <= i < base**n, 'i should be in range [0,base**n)!'
    vec = [0]*n
    rem = i
    for j in range(n):
        vec[j], rem = divmod(rem,2**(n-j-1))
    return vec
#------------------------------------------------------------------------------
def vec2real(v):
    '''
    Convertion from real occupation state to index of many-body state
    
    Return index of many-body state.
    
    Parameters
    ----------
    v : [n] vector
        Array representing the occupation of each site
        
    Returns
    -------
    out: int
        Index of many-body state
    
    Examples
    --------
    >>> vec2real([0,0,1,0])
    >>> 2

    >>> vec2real([1,0,0,0])
    >>> 8

    >>> vec2real([1,0,1,0])
    >>> 10
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
#------------------------------------------------------------------------------
def redmatl(v,l):
    '''
    Reduced density matrix of left sub-system
    
    Return reduced density matrix.
    
    Parameters
    ----------
    v : [n] vector
        Pure state in Hilbert space
    l : int
        Number of particles of left subsystem
    
    Returns
    -------
    out : [l,l] matrix
        Reduced density matrix
    '''
    assert isinstance (v,(list,tuple,np.ndarray)), 'v must be vector!'
    assert isinstance (l,int), 'n must be integer!'
    assert l <= MAT_MAX, 'l is to large!'
    m = v.reshape([base**l,-1],order='c')
    out = m@m.conj().T
    return out
#------------------------------------------------------------------------------
def redmatr(v,l):
    '''
    Reduced density matrix of right sub-system
    
    Return reduced density matrix.
    
    Parameters
    ----------
    v : [n] vector
        Pure state in Hilbert space
    l : int
        Number of particles of right subsystem
    
    Returns
    -------
    out : [l,l] matrix
        Reduced density matrix
    '''
    assert isinstance (v,(list,tuple,np.ndarray)), 'v must be vector!'
    assert isinstance (l,int), 'n must be integer!'
    assert l <= MAT_MAX, 'l is to large!'
    m = v.reshape([-1,base**l],order='f')
    out = m@m.conj().T
    return out
#------------------------------------------------------------------------------
def redmatm(v,start,l):
    '''
    Reduced density matrix of middle sub-system
    
    Return reduced density matrix.
    
    Parameters
    ----------
    v : [n] vector
        Pure state in Hilbert space
    start : int
        Start index of the subsystem
    l : int
        Number of particles of middle subsystem
        
    Returns
    -------
    out : [l,l] matrix
        Reduced density matrix
    '''
    assert isinstance (v,(list,tuple,np.ndarray)), 'v must be vector!'
    assert isinstance (l,int), 'n must be integer!'
    assert l <= MAT_MAX, 'l is to large!'
    m = v.reshape([base**(start+l),-1]).reshape([base**l,-1],order='f')
    out = m@m.conj().T
    return out
#------------------------------------------------------------------------------
def redmato(v,left,right):
    '''
    Reduced density matrix of outer sub-system.
    
    Return reduced density matrix.
    
    Parameters
    ----------
    v : [n] vector
        Pure state in Hilbert space
    left : int
        Number of left particles
    right : int
        Number of right particles
        
    Returns
    -------
    out : [left+right,left+right] matrix
        Reduced density matrix
    '''
    assert isinstance (v,(list,tuple,np.ndarray)), 'v must be vector!'
    assert isinstance (left,int), 'left must be integer!'
    assert isinstance (right,int), 'right must be integer!'
    assert left+right <= MAT_MAX, 'left+right is to large!'
    m = v.reshape([base**left,-1],order='f').reshape([base**(left+right),-1])
    out = m@m.conj().T
    return out
#------------------------------------------------------------------------------
def entropy(rho):
    '''
    Entanglement entropy
    
    Return entanglement entropy of reduced density matrix.
    
    Parameters
    ----------
    rho : [n,n] matrix
        Reduced density matrix
            
    Returns
    -------
    out : float
        Entanglement entropy
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
    Number of particles
    
    Return number of particles from a vector in Hilbert space
    
    Parameters
    ----------
    v : [n] vector
        Reduced density matrix
            
    Returns
    -------
    n : int
        Number of particles
    '''
    assert isinstance (v,(list,tuple,np.ndarray)), 'v must be vector!'
    n = int(math.log(len(v),base))
    return n
#------------------------------------------------------------------------------
def mesure(v):
    '''
    Average of occupation of each quantum state on each site
    
    Return expectation of each quantum state of each particle.
    
    Parameters
    ----------
    v : [...,n] vector/matrix
        Vector (stack of vectors) of Hilbert space
        
    Returns
    -------
    out : [...,n,l] array
        Expectation of each quantum state of each particle
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
###############################################################################
# State evolement routine
#------------------------------------------------------------------------------
def evo(H,v,t):
    '''
    Evolving vector in Herbert space
    
    Return evolved vector in Hilbert space after time t.
    
    Parameters
    ----------
    H : [n,n] (sparse) matrix
        Hamiltonion
    v : [n] vector/matrix
        Vector of Hilbert space
    t : float
        Evolving time
    
    Returns
    -------
    out : [n] vector
        Vector after evolving time t
    '''
    assert isinstance (H,(np.ndarray,sparse.spmatrix)), 'v must be numpy matrix or sparse matrix!'
    assert isinstance (v,np.ndarray), 'v must be vector!'
    assert H.shape == (len(v),)*2, 'H must be of the same length as v!'
    assert t.imag == 0, 't must be real!'
    out = linalg.expm_multiply(-1j*H*t,v)
    return out
#------------------------------------------------------------------------------
def evom(H,v,start,end,num):
    '''
    Multiple vector evolving
    
    Return stack of vectors in Hilbert space after multiple evolving times.
    
    Parameters
    ----------
    H : [n,n] (sparse) matrix
        Hamiltonion
    v : [n] vector/matrix
        Vector of Hilbert space
    start : float
        Start time
    end: float
        End time
    num: int
        Number of division
            
    Returns
    -------
    out: [num,n] matrix
        Stack of vector after multiple evolving times
    '''
    assert isinstance (H,(np.ndarray,sparse.spmatrix)), 'v must be numpy matrix or sparse matrix!'
    assert isinstance (v,np.ndarray), 'v must be vector!'
    assert H.shape == (len(v),)*2, 'H must be of the same length as v!'
    assert start.imag == 0 , 'start must be real!'
    assert end.imag == 0 , 'end must be real!'
    out = linalg.expm_multiply(-1j*H,v,start=start,stop=end,num=num)
    return out
###############################################################################
# Spin specifi routine
# valid only if base = 2
# Creation
#------------------------------------------------------------------------------
def sx(i,n):
    '''
    Sx matrix in many-body space
    
    Return kronecker product of Pauli Sx matrix on site i and identity operator
    on other sites.
    
    Parameters
    ----------
    i : int
        Index of the operator
    n : int
        Number of particles
            
    Returns
    -------
    out: [base**n,base**N] sparse matrix
        Operator in whole space
    '''
    assert isinstance (i,int), 'i must be integer!'
    assert isinstance (n,int), 'n must be integer!'
    assert n <= SPARSE_MAX, 'n is to large!'
    assert 0 <= i <= n, 'i should be in range [0,n]!'
    s = sparse.coo_matrix(([1,1],([0,1],[1,0])),dtype=dtype)
    out = opn(s,i,n)
    return out
#------------------------------------------------------------------------------
def sy(i,n):
    '''
    Sy matrix in many-body space
    
    Return kronecker product of Pauli Sy matrix on site i and identity operator
    on other sites.
    
    Parameters
    ----------
    i : int
        Index of the operator
    n : int
        Number of particles
            
    Returns
    -------
    out: [base**n,base**N] sparse matrix
        Operator in whole space
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
#------------------------------------------------------------------------------
def sz(i,n):
    '''
    Sz matrix in many-body space
    
    Return kronecker product of Pauli Sz matrix on site i and identity operator
    on other sites.
    
    Parameters
    ----------
    i : int
        Index of the operator
    n : int
        Number of particles
            
    Returns
    -------
    out: [base**n,base**N] sparse matrix
        Operator in whole space
    '''
    assert isinstance (i,int), 'i must be integer!'
    assert isinstance (n,int), 'n must be integer!'
    assert n <= SPARSE_MAX, 'n is to large!'
    assert 0 <= i <= n, 'i should be in range [0,n]!'
    s = sparse.coo_matrix(([1,-1],([0,1],[0,1])),dtype=dtype)
    out = opn(s,i,n)
    return out
#------------------------------------------------------------------------------
# Spin chain Hamiltonian
#------------------------------------------------------------------------------
def Hspin(jx=0,jy=0,jz=0,hx=0,hy=0,hz=0,**arg):
    '''
    Hamiltonian for nearest interaction spin chain with transverse field.
    
    Return sparse matrix for many-body Hamiltonian.
    
    Note this function change the module's global parameter:
        base -> 2
        dtype -> float
    
    Parameters
    ----------
    jx : float/[n] array, optional
        Spin interaction in x direction
    jy : float/[n] array, optional
        Spin interaction in y direction
    jz : float/[n] array, optional
        Spin interaction in z direction
    hx : float/[n] array, optional
        Transverse field in x direction
    hy : float/[n] array, optional
        Transverse field in y direction
    hz : float/[n] array, optional
        Transverse field in z direction
    
    Returns
    -------
    H: [2**n,2**n] sparse matrix
        Many-body Hamiltonian
    '''
    global dtype
    # assign data type
    try:
        dtype = arg['dtype']
    except KeyError:
        dtype = 'f4'
    # assign lenth
    try:
        n = arg['n']
    # find length from given parameter
    except KeyError:
        try:
            n = len(jx)
        except TypeError:
            try:
                n = len(jy)
            except TypeError:
                try:
                    n = len(jz)
                except TypeError:
                    try:
                        n = len(hx)
                    except TypeError:
                        try:
                            n = len(hy)
                        except TypeError:
                            try:
                                n = len(hz)
                            except TypeError:
                                n = 0
    try:
        len(jx)
    except TypeError:
        jx = np.full(n,jx)
    try:
        len(jy)
    except TypeError:
        jy = np.full(n,jy)
    try:
        len(jz)
    except TypeError:
        jz = np.full(n,jz)
    try:
        len(hx)
    except TypeError:
        hx = np.full(n,hx)
    try:
        len(hy)
    except TypeError:
        hy = np.full(n,hy)
    try:
        len(hz)
    except TypeError:
        hz = np.full(n,hz)
    H = sparse.coo_matrix((2**n,2**n),dtype=dtype)
    Sx = [sx(i,n) for i in range(n)]
    Sy = [sy(i,n) for i in range(n)]
    Sz = [sz(i,n) for i in range(n)]
    for i in range(n-1):
        H += +jx[i]*Sx[i]@Sx[i+1] \
             -jy[i]*Sy[i]@Sy[i+1] \
             +jz[i]*Sz[i]@Sz[i+1] \
             +hx[i]*Sx[i] \
             +hy[i]*Sy[i] \
             +hz[i]*Sz[i]
    H += +jx[n-1]*Sx[n-1]@Sx[0] \
         -jy[n-1]*Sy[n-1]@Sy[0] \
         +jz[n-1]*Sz[n-1]@Sz[0] \
         +hx[n-1]*Sx[n-1] \
         +hy[n-1]*Sy[n-1] \
         +hz[n-1]*Sz[n-1]
    return H
#------------------------------------------------------------------------------
# Mesurement
#------------------------------------------------------------------------------
def spin(v):
    '''
    Spin messuremeent
    
    Return expectation of spin of each particle.
    
    Parameters
    ----------
    v : [...,n] vector/matrix
        Vector (stack of vectors) of Hilbert space
        
    Returns
    -------
    out : [...,n,3] array
        Expectation of spin of each particle
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
#------------------------------------------------------------------------------
def spinz(v):
    '''
    Spin-z messuremeent
    
    Return expectation of spin-z of each particle.
    
    Parameters
    ----------
    v : [...,n] vector/matrix
        Vector (stack of vectors) of Hilbert space
        
    Returns
    -------
    out : [...,n,3] array
        Expectation of spin-z of each particle
    '''
    temp = mesure(v)
    out = -np.diff(temp)
    return out
#------------------------------------------------------------------------------
# Spin chain class
#------------------------------------------------------------------------------
class spinchain:
    '''
    Class for many-body state of spin chain
    '''
    def __init__(self,H=None):
        '''
        Initiate the spin chain state
        
        Parameters
        ----------
        H : [n,n] (sparse) matrix, optional
            Spin chain Hamiltonian
        '''
        if H:
            self.H = H
            self.dim = H.shape[0]
            self.site = int(math.log2(self.dim))
    #--------------------------------------------------------------------------
    def __call__(self,jx=0,jy=0,jz=0,hx=0,hy=0,hz=0,**arg):
        '''
        Create a spin chain Hamiltonian for the class object
        
        Parameters
        ----------
        jx : float/[n] array, optional
            Spin interaction in x direction
        jy : float/[n] array, optional
            Spin interaction in y direction
        jz : float/[n] array, optional
            Spin interaction in z direction
        hx : float/[n] array, optional
            Transverse field in x direction
        hy : float/[n] array, optional
            Transverse field in y direction
        hz : float/[n] array, optional
            Transverse field in z direction
        '''
        self.H = Hspin(jx,jy,jz,hx,hy,hz,**arg)
        self.dim = self.H.shape[0]
        self.site = int(math.log2(self.dim))
    #--------------------------------------------------------------------------
    def setvec(self,v,kink=False):
        '''
        Set the vector to be an product state
        
        Parameters
        ----------
        v : int
            Index of many-body basis
        kink: bool/int/list, optional
            Give a kink state if given a list
        '''
        if kink:
            v = np.zeros(self.site,dtype=int)
            v[kink] = 1
            site = vec2real(v)
            vec = np.zeros(2**self.dim)
            vec[site] = 1
            self.vec = vec
        else:
            site = vec2real(v)
            vec = np.zeros(2**self.dim)
            vec[site] = 1
            self.vec = vec
    #--------------------------------------------------------------------------
    def rand(self):
        '''
        Give a random state
        '''
        vr = np.random.rand(self.dim)-0.5
        vi = np.random.rand(self.dim)-0.5
        v = vr + 1j*vi
        self.vec = v/np.linalg.norm(v)
    #--------------------------------------------------------------------------
    def spin(self):
        '''
        Return spin expectation of each site
        '''
        out = spin(self.vec)
        return out
    #--------------------------------------------------------------------------
    def spinz(self):
        '''
        Return spin-z expectation of each site
        '''
        out = spinz(self.vec)
        return out
    #--------------------------------------------------------------------------
    def evo(self,t):
        '''
        Evolve the state
        
        Parameters
        ----------
        t : float
            Evolving time
        '''
        self.vec = linalg.expm_multiply(-1j*self.H*t,self.vec)
    #--------------------------------------------------------------------------
    def evom(self,ts,te,n):
        '''
        State evolving for multiple time
        
        Parameters
        ----------
        ts : float
            Start time
        te : float
            End time
        n: int
            Time division
        
        Returns
        -------
        out : [n,dim] matrix
            Stack of vector after multiple evolving times
        '''
        out = linalg.expm_multiply(-1j*self.H,self.vec,start=ts,stop=te,num=n)
        self.vec = out[-1]
        return out
    #--------------------------------------------------------------------------
    def redmat(self,*arg,sec='left'):
        '''
        Reduced density matrix
        
        Parameters
        ----------
        *arg : list of arguments
            Section specific arguments, see reduced matrix part
        sec : string, optional
            Section of sub-system.
            Choose from ['left','right','middle','outer']
            
        Returns
        -------
        out : [m,m] matrix
            Reduced density matrix
        '''
        if sec == 'left':
            out = redmatl(self.vec,*arg)
        elif sec == 'right':
            out = redmatr(self.vec,*arg)
        elif sec == 'middle':
            out = redmatm(self.vec,*arg)
        elif sec == 'outer':
            out = redmato(self.vec,*arg)
        else:
            print('illegal section for reduced matrix!')
            sys.exit()
        return out
    #--------------------------------------------------------------------------
    def S_ent(self,*arg,sec='left'):
        '''
        Entanglement entropy
        
        Parameters
        ----------
        *arg : list of arguments
            Section specific arguments, see reduced matrix part
        sec : string, optional
            Section of sub-system.
            Choose from ['left','right','middle','outer']
            
        Returns
        -------
        S : float
            Entanglement entropy
        '''
        rho = self.redmat(*arg,sec)
        S = entropy(rho)
        return S
###############################################################################
# Fermion specific routine
# valid only if base = 2
# Creation
#------------------------------------------------------------------------------
def fp(i,n):
    '''
    Fermion creation matrix in many-body space
    
    Return kronecker product of Fermion creation operator on site i and identity 
    operator on other sites.
    
    Parameters
    ----------
    i : int
        Index of the operator
    n : int
        Number of particles
            
    Returns
    -------
    out: [2**n,2**n] sparse matrix
        Operator in whole space
    '''
    assert isinstance (i,int), 'i must be integer!'
    assert isinstance (n,int), 'n must be integer!'
    assert n <= SPARSE_MAX, 'n is to large!'
    assert 0 <= i <= n, 'i should be in range [0,n]!'
    p = sparse.coo_matrix(([1],([0],[1])),shape=[2,2],dtype=dtype)
    s = sparse.coo_matrix(([-1,1],([0,1],[0,1])),dtype=dtype)
    if i == 0 or i == n:
        return sparse.kron(p,idn(n-1),format='coo')
    out = sparse.coo_matrix(([-1,1],([0,1],[0,1])),dtype=dtype)
    for j in range(i-1):
        out = sparse.kron(out,s,format='coo')
    out = sparse.kron(out,p,format='coo')
    out = sparse.kron(out,idn(n-i-1),format='coo')
    return out
#------------------------------------------------------------------------------
def fm(i,n):
    '''
    Fermion annihilation matrix in many-body space
    
    Return kronecker product of Fermion annihilation operator on site i and 
    identity operator on other sites.
    
    Parameters
    ----------
    i : int
        Index of the operator
    n : int
        Number of particles
            
    Returns
    -------
    out: [2**n,2**n] sparse matrix
        Operator in whole space
    '''
    assert isinstance (i,int), 'i must be integer!'
    assert isinstance (n,int), 'n must be integer!'
    assert n <= SPARSE_MAX, 'n is to large!'
    assert 0 <= i <= n, 'i should be in range [0,n]!'
    p = sparse.coo_matrix(([1],([1],[0])),shape=[2,2],dtype=dtype)
    s = sparse.coo_matrix(([-1,1],([0,1],[0,1])),dtype=dtype)
    if i == 0 or i == n:
        return sparse.kron(p,idn(n-1),format='coo')
    out = sparse.coo_matrix(([-1,1],([0,1],[0,1])),dtype=dtype)
    for j in range(i-1):
        out = sparse.kron(out,s,format='coo')
    out = sparse.kron(out,p,format='coo')
    out = sparse.kron(out,idn(n-i-1),format='coo')
    return out
#------------------------------------------------------------------------------
def fn(i,n):
    '''
    Fermion occupation matrix in many-body space
    
    Return kronecker product of Fermion occupation operator on site i and 
    identity operator on other sites.
    
    Parameters
    ----------
    i : int
        Index of the operator
    n : int
        Number of particles
            
    Returns
    -------
    out: [2**n,2**n] sparse matrix
        Operator in whole space
    '''
    assert isinstance (i,int), 'i must be integer!'
    assert isinstance (n,int), 'n must be integer!'
    assert n <= SPARSE_MAX, 'n is to large!'
    assert 0 <= i <= n, 'i should be in range [0,n]!'
    nn = sparse.coo_matrix(([1],([0],[0])),shape=[2,2],dtype=dtype)
    out = opn(nn,i,n)
    return out
#------------------------------------------------------------------------------
def Hfermion(A,B,C=None,**kwd):
    '''
    Hamiltonian for fermion model with hopping, pairing and density interaction
    
    Parameters
    ----------
    A : [n,n] matrix
        Hopping matrix
    B : [n,n] matrix
        pairing matrix
    C : [n,n] matrix
        Interaction matrix
    **kwd: str, keyword
        'check' : bool
            Whether do the check
            
    Returns
    -------
    H: [2**n,2**n] sparse matrix
        Operator in whole space
    '''
    check = True
    try:
        if kwd['check']:
            check = False
    except KeyError:
        pass
    if check:
        assert len(A) == len(B) and len(B)<=MAT_MAX, 'Illegal dimension!'
        assert np.sum(np.abs(A-A.T)) < 1.e-10, 'A must be symmetric!'
        assert np.sum(np.abs(B+B.T)) < 1.e-10, 'B must be anti-symmetric!'
    n = len(A)
    H = empty(n)
    Fp = [fp(i,n) for i in range(n)]
    Fm = [fm(i,n) for i in range(n)]
    for i in range(n):
        for j in range(n):
            if not A[i,j]==0:
                H += A[i,j]*Fp[i]@Fm[j]
            if not B[i,j]==0:
                H += B[i,j]*(Fp[i]@Fp[j] - Fm[i]@Fm[j])
    if C:
        Fn = [fn(i,n) for i in range(n)]
        for i in range(n):
            for j in range(n):
                if not C[i,j]==0:
                    H += C[i,j]*Fn[i]@Fn[j]
    return H
###############################################################################
# test module
if __name__ == '__main__':
    def tfn():
        a = fn(0,2)
        b = fn(1,2)
        c = fn(1,3)
        return a,b,c
    def thfer():
        A = np.random.rand(10,10)
        B = np.random.rand(10,10)
        A = A+A.T
        B = B-B.T
        h = Hfermion(A,B)
        return h