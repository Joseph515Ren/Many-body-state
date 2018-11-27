###############################################################################
'''
Module for XXZ model
11/19/2018
'''
###############################################################################
import numpy as np
import math
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt
import manybodylib as mb
from manybodylib import sx,sy,sz
###############################################################################
# XXZ Hamiltonian
mb.dtype = 'f4'
def xxz(jx=1,jz=1,v=0,n=12,sp=True):
    '''
    return XXZ Hamiltonian
    input:
        jx: float, optional
            jx intensity
        jz: float, optional
            jy intensity
        v: float, optional
            disorder intensity
        n: int, optional
            number of site
        sp: bool, optional
            return sparse matrix or not
    output:
        out: [2**n,2**n] matrix
            XXZ Hamiltonian
    '''
    h = 2*v*(np.random.rand(n)-0.5)
    H = sparse.coo_matrix((2**n,2**n),dtype='f4')
    for i in range(n):
        H +=-jx*sx(i,n)@sx(i+1,n) \
            +jx*sy(i,n)@sy(i+1,n) \
            -jz*sz(i,n)@sz(i+1,n) \
            -h[i]*sz(i,n)
    if sp:
        return H
    else:
        return H.toarray()
###############################################################################
# spin chain class
class chain:
    '''
    class for many-body state of XXZ model
    '''
    def __init__(self,H):
        '''
        initiate the xxz chain state
        '''
        self.H = H
        self.length = H.shape[0]
        self.site = int(math.log2(self.length))
        self.vec = np.zeros(self.length,dtype=complex)
    def kink(self,i):
        '''
        give a kink state
        i can be a list
        '''
        v = np.zeros(self.site,dtype=int)
        v[i] = 1
        site = mb.vec2real(v)
        self.vec[site] = 1
    def rand(self):
        '''
        give a random state
        '''
        vr = np.random.rand(self.length)-0.5
        vi = np.random.rand(self.length)-0.5
        v = vr + 1j*vi
        self.vec = v/np.linalg.norm(v)
    def reset(self):
        '''
        reset the chain state
        '''
        self.vec = np.zeros(self.length,dtype=complex)
    def spin(self):
        '''
        return spin z expectation of each particle
        '''
        return np.diff(mb.mesure(self.vec,self.site))[:,0]
    def evo(self,t):
        '''
        evolve the state
        '''
        self.vec = linalg.expm_multiply(-1j*self.H*t,self.vec)
    def evo_mul(self,ts,te,n):
        '''
        multi evolve the state
        return the vectors
        '''
        out = linalg.expm_multiply(-1j*self.H,self.vec,start=ts,stop=te,num=n)
        self.vec = out[-1]
        return out
    def rho(self,i):
        '''
        return reduced density matrix
        '''
        return mb.redmatl(self.vec,i,self.site)
    def S_ent(self,i):
        '''
        return entanglement entropy
        '''
        r = self.rho(i)
        return mb.entropy(r)
    def ent_spec(self,i,plot=False):
        '''
        return entanglement spectrum
        '''
        r = self.rho(i)
        e, v = np.linalg.eigh(r)
        e = e[e>0]
        n = len(e)
        z = mb.spinz(v[:,:n].T).sum(1)
        if plot:
            plt.scatter(z,e,marker='^')
            plt.show()
            return z,e
        else:
            return z,e
    def psi4(self):
        '''
        return |psi|^4
        '''
        return np.sum(np.abs(self.vec)**4)
###############################################################################
# test
if __name__ == '__main__':
    pass