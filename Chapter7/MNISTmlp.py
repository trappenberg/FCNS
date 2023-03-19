import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('mnist_784.csv')
X = df.iloc[:,0:784].values/255
Y1 = df.iloc[:,784].values

Y=np.zeros((10,70000))
for i in range(70000): Y[Y1[i],i]=1
X=np.append(X.T,np.array([np.ones(70000)]),axis=0)


# model specifications
Ni=785; Nh=128; No=10

#parameter and array initialization
Ntrials=100
wh=np.random.randn(Nh,Ni); dwh=np.zeros(wh.shape) 
wo=np.random.randn(No,Nh); dwo=np.zeros(wo.shape) 
error=np.array([])

for trial in range(Ntrials):
    i=np.random.permutation(range(1000))
    for batch in range(10):
        Xtrain=X[:,batch*100:(batch+1)*100]
        Ytrain=Y[:,batch*100:(batch+1)*100]
        
        h=wh@Xtrain; h=np.where(h<0,0.1*h,h)
        y=np.exp(wo@h); y=y/np.sum(y,0)
        do=(Ytrain-y)  
        dh=wo.T@do  
    
        # update weights with momentum
        dwo=0.9*dwo+do@h.T
        wo=wo+0.001*dwo
        dwh=0.9*dwh+dh@Xtrain.T
        wh=wh+0.001*dwh
    
        error=np.append(error,np.sum(abs(Ytrain-y)))
plt.plot(error)
