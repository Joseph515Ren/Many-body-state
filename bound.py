###############################################################################
'''
Lieb-Robinson bound & OTOC
11/19/2018
'''
###############################################################################
import numpy as np
from manybodylib import sz
from ising import xxz
###############################################################################
workdir = './test'  # working diretory

n = 12      # number of particles
v = 1       # disorder intensity
center = 2  # center site
start = 0   # start time
end = 1.2   # end time
nt = 100    # time division
###############################################################################
def lbotoc(A,e,dt):
    '''
    Calculate Lieb-Robinson bound & OTOC
    '''
    lb = np.empty(n)
    otoc = np.empty(n)
    d1 = A[center]*np.exp(1j*dt*e)
    for i in range(n):
        d2 = A[i]*np.exp(-1j*dt*e)
        m = d1@d2
        m = m - m.conj().T
        psi = m[:,0]
        lb[i] = np.linalg.norm(m)
        otoc[i] = np.real(psi.conj()@psi)
    return lb, otoc
###############################################################################
def main():
    '''main function'''
    global v,n
    H = xxz(v=v,n=n).toarray()
    e, vec = np.linalg.eigh(H)
    H = 0
    A = []
    for i in range(n):
        A.append(vec.T@sz(i,n)@vec)
    t = np.linspace(start,end,nt)
    lb = np.empty([nt,n])
    otoc = np.empty([nt,n])
    for i in range(nt):
        lb[i], otoc[i] = lbotoc(A,e,t[i])
        print('finish %d/%d' %(i+1,nt))
    np.savetxt('%s/lb.dat' % workdir,lb)
    np.savetxt('%s/otoc.dat' % workdir,otoc)
###############################################################################
if __name__=='__main__':
    main()