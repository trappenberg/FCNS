import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('mnist_784.csv')
X = df.iloc[:,0:784].values/255
X=np.append(X.T,np.array([np.ones(70000)]),axis=0)

# model specifications
N0=785; N1=128

#parameter and array initialization
Ntrials=100
w=np.random.randn(N1,N0)
error=np.array([])

for trial in range(Ntrials):
    e=0
    for i in range(100):
        r=np.random.randn(N1)
        for t in range(10):
            dI=X[:,i]-w.T@r/N1
            r=r+0.1*w@dI.T
        w=w+0.1*np.outer(r,dI)             
        e=e+dI@dI
    error=np.append(error,e)
plt.plot(np.sqrt(error))
