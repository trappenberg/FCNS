import numpy as np
import matplotlib.pyplot as plt

gamma=0
w=np.zeros(10); 
Z=np.zeros((100,6)); rhatRec=np.zeros(100)
x=np.random.rand(10,6)
r=np.array([0,0,1,0,0.5,0])

for trial in range(100):
    V=0; 
    for t in range(1,6):
        Vlast=V
        V=w@x[:,t]
        rhat=r[t-1]+gamma*V-Vlast
        w=w+0.1*rhat*x[:,t-1]        
        Z[trial,t-1]=Vlast
        rhatRec[trial]=rhatRec[trial]+rhat
plt.figure(); plt.plot(Z[-1,:])
plt.figure(); plt.plot(rhatRec/5)
