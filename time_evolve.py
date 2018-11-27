import numpy as np
from scipy import sparse
from ising import xxz, chain
###############################################################################
workdir = './test'  # working diretory

dt = 0.01  # time precision
nt = 30    # number of evolvement each batch
N = 3       # number of evolement batch

v = 1       # disorder intensity
n = 10      # number of particles
kink = 4   # kink site
###############################################################################
def evo(i):
    '''evolving method'''
    h = sparse.load_npz('%s/h.npz' % workdir)
    c = chain(h)
    c.vec = np.load('%s/%d.npy' %(workdir,i))
    log = c.evo_mul(dt,(nt+1)*dt,nt)
    for j in range(nt):
        np.save('%s/%d.npy' %(workdir,j+1+i),log[j])
    return
###############################################################################
def main():
    '''main function'''
    h = xxz(v=v,n=n)
    sparse.save_npz('%s/h' % workdir,h)
    c = chain(h)
    c.kink(kink)
    np.save('%s/0.npy' % workdir,c.vec)
    for i in range(N):
        evo(i*nt)
        print('finish %d/%d' %(i+1,N))
###############################################################################  
if __name__=='__main__':
    main()