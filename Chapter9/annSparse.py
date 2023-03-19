# Sparse auto-associative network
import numpy as np
import matplotlib.pyplot as plt

nn=500; a=0.1; npat=40
aret=np.zeros((20,10))
hd=np.zeros((20,10))

for irun in range(10):
    pat=np.zeros((nn,npat));
    for i in range(npat): 
        idx=np.random.permutation(nn)
        pat[idx[:int(a*nn)],i]=1   
    w=(pat-a)@(pat-a).T; w=w/np.sqrt(npat); ic=-1
    sparsity=np.arange(0,0.2,0.01)
    for c in sparsity:
        ic=ic+1
        s=pat[:,0]; s[:10]=1-s[:10]
        for t in range(10): s=((w-c)@s)>0 
        aret[ic,irun]=np.sum(s)/nn
        hd[ic,irun]=((1-s).T@pat[:,0]+s.T@(1-pat[:,0]))/nn
plt.errorbar(sparsity,np.mean(aret,axis=1),yerr=np.std(aret,axis=1))
plt.errorbar(sparsity,np.mean(hd,axis=1),yerr=np.std(hd,axis=1))
