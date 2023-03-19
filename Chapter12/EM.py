# 1d example EM algorithm
import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, mu, var):
    return np.exp(-(x-mu)**2/(2*var))/np.sqrt(2*np.pi*var)

x0=np.arange(-10,10,0.1)
var1=1; var2=1; mu1=-2; mu2=2;
for  iteration in range(10):
    # plot distribution
    plt.figure()
    plt.plot(x0, gaussian(x0,-1,4),'k--')
    plt.plot(x0, gaussian(x0,4,.25),'k:')
    plt.plot(x0, gaussian(x0,mu1,var1),'r')
    plt.plot(x0, gaussian(x0,mu2,var2),'b')
    # data
    x=np.array([np.random.normal(-1,2,50),np.random.normal(4,0.5,50)])
    # expectation (recognition)
    c=gaussian(x,mu1,var1)>gaussian(x,mu2,var2)
    # maximization
    mu1=np.sum(x[c>0.5])/np.sum(c)
    var1=np.sum((x[c>0.5]-mu1)**2)/np.sum(c)
    mu2=np.sum(x[c<0.5])/(100-np.sum(c))
    var2=np.sum((x[c<0.5]-mu2)**2)/(100-np.sum(c))
