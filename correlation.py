###############################################################################
'''
Correlation function
11/19/2018
'''
###############################################################################
import numpy as np
import manybodylib as mb
###############################################################################
loaddir = './1.TH/evo'      # loading directory
savedir = './1.TH/res'          # saving directory
num = 600                   # number of data

n = 22                      # particle number
center = 10                 # center site
###############################################################################
def cor(i,Z):
    '''
    calculate correlation function of Sz using existing data
    '''
    out = np.empty(n)
    v = np.load('%s/%d.npy' %(loaddir,i))
    vl = []
    si = []
    for i in range(n):
        vl.append(Z[i]@v)
    for i in range(n):
        si.append(v.conj()@vl[i])
    for i in range(n):
        sisj = vl[i].conj()@vl[center]
        out[i] = np.abs(sisj-si[i]*si[center])
    return out
###############################################################################
def main():
    '''main function'''
    mb.dtype = 'f4'
    Z = []
    for i in range(22):
        Z.append(mb.sz(i,n))
    log = np.empty([num,n])
    for i in range(num):
        log[i] = cor(i,Z)
        print('%d/%d' %(i+1,num))
    np.savetxt('%s/correlation.dat' % savedir,log)
    
###############################################################################
if __name__=='__main__':
    main()