###############################################################################
'''
Phase of entanglement entropy
11/19/2018
'''
###############################################################################
import numpy as np
from scipy import sparse
from manybodylib import sx,sy,sz
from manybodylib import redmatl, entropy
###############################################################################
workdir = './test'      # working directory

n = 8       # number of particles
jx = 1      # jx strength
jz = 1      # jz strength
l = 4       # number of particles in subsystem
v = np.linspace(0,7,10)     # disorder strength
###############################################################################
def xxz(v,ran):
    '''
    return XXZ Hamiltonian for specific random field
    input:
        v: float
            disorder intensity
        ran: [n] vector
            disorder vector
    output:
        H: [n,n] matrix
            Haniltonian
    '''
    global jx,jy,n
    h = v*ran
    H = sparse.coo_matrix((2**n,2**n),dtype='f4')
    for i in range(n):
        for i in range(n):
            H +=-jx*sx(i,n)@sx(i+1,n) \
                +jx*sy(i,n)@sy(i+1,n) \
                -jz*sz(i,n)@sz(i+1,n) \
                -h[i]*sz(i,n)
    return H.toarray()
###############################################################################
def analyse(e,v,l):
    '''
    return entanglement entropy of each eigenstate
    input:
        e: [n] vector
            ordered eigenvalues
        v: [n,n] matrix
            ordered eigenvectors
        l:
            number of particles in subsystem
    output:
        re: [n] vector
            renormalized energy ranges in (0,1)
        s: [n] vector
            entanglement entropy of each eigenstate
    '''
    dE = e[-1]-e[0]
    re = (e-e[0])/dE
    s = np.empty_like(e)
    for i in range(len(e)):
        rho = redmatl(v[:,i],l)
        s[i] = entropy(rho)
    return re,s
###############################################################################
def scanv(v,ran):
    '''
    scan the entanglement entropy along give array of v
    input:
        v: [nv] vector
            array of v
        ran: [n] vector
            disorder vector
    output:
        vout: [2**n,nv] matrix
            meshgid matrix for v
        eout: [2**n,nv] matrix
            meshgid matrix for e
        mout: [2**n,nv] matrix
            meshgid matrix for entanglement entropy
    '''
    global n,l
    nv = len(v)
    eout = np.empty([2**n,nv])
    vout = np.tile(v,(2**n,1))
    mout = np.empty_like(eout)
    for i in range(nv):
        h = xxz(v[i],ran)
        e,vv = np.linalg.eigh(h)
        eout[:,i],mout[:,i] = analyse(e,vv,l)
        print('%d/%d' %(i+1,nv))
    return vout,eout,mout
###############################################################################
def main():
    '''main function'''
    global v
    ran = 2*(np.random.rand(n)-0.5)
    vo,eo,mo = scanv(v,ran)
    np.savetxt('%s/v.dat' % workdir,vo)
    np.savetxt('%s/e.dat' % workdir,eo)
    np.savetxt('%s/s.dat' % workdir,mo)
###############################################################################
if __name__=='__main__':
    main()