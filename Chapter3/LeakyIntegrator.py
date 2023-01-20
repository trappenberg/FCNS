import numpy as np
import matplotlib.pyplot as plt

tmax=10; dt=1; tau=2
X = np.zeros(tmax*int(1/dt)); 
a=np.array([1,3,8])/dt; X[a.astype(int)]=1

x = np.array([0]); 
for t in range(0,tmax*int(1/dt)):
    x = np.append(x,x[t]+dt*(-1/tau*x[t])+X[t])
    
plt.plot(x); 
plt.xlabel('t in units of dt'); plt.ylabel('x(t)')
plt.savefig('li1.pdf')