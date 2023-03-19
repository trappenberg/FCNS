# Weight distribution of Hebbian synapses in rate model
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

nn=100; npat=1000 # number of nodes and patterns

rPre = np.random.exponential(10,(nn,npat)) 
rPost = np.random.exponential(20,(nn,npat)) 
w=(rPost-np.mean(rPost))@(rPre--np.mean(rPre)).T/npat

# Histogram plotting
n, bins = np.histogram(w) 
dx = bins[1]-bins[0]
n = n/(sum(n)*dx)
x = np.zeros(10)
for i in range(10):
    x[i] = bins[i]+ dx/2
plt.bar(x,n,8)

# Fit normal ditribution to data
def func(x, mu, sig):
  return 1/(np.sqrt(2*np.pi)*sig)*np.exp(-(x-mu)**2/(2*sig**2))

popt, pcov = curve_fit(func, x, n)
x = np.linspace(x[0],x[9],50)
plt.plot(x, func(x, *popt), 'r-')
plt.xlabel('w'), plt.ylabel('P(w)')
