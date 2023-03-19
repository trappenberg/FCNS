import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#sigmoid activation fcn
sig = lambda x: 1 / (1 + np.exp(-x)) 

#data selection
df = pd.read_csv('mnist_784.csv')
X = df.iloc[:,0:784].values/255

ndata=10; nhidden=100;  nvisible=28*28;  nepochs=200 
e=0.01;  noise=0.05;  ngibbs=3;  T=1/4
w= 0.1*np.random.randn(nvisible,nhidden) 
vbias= np.zeros(nvisible);  hbias= np.zeros(nhidden)  

X = np.array([X[34,:],X[8,:],X[5,:],X[7,:],X[9,:],
              X[0,:],X[32,:],X[15,:],X[17,:],X[22,:]])

# RBM training 
err = np.zeros(nepochs)
for epoch in range(nepochs): 
  for v in X:
    h  = sig(v @ w + hbias)
    hsample = h > np.random.rand(nhidden)      
    vrecon = sig(w @ hsample + vbias)
    hrecon = sig(vrecon @ w + hbias)
    w += e*(np.outer(v,h) - np.outer(vrecon,hrecon))  
    hbias+= e*(h-hrecon);  vbias+= e*(v-vrecon)
    err[epoch] = ((v-vrecon)**2).sum()
plt.plot(err,'.'); plt.xlabel('epoch'); plt.ylabel('error')

# Plot interations of noisy patterns
r = np.random.rand(ndata,nvisible) < noise
flipped = (1-r)*X + r*(1.-X)  #flip random bits in input
plt.figure() 
for g in range(ngibbs):
  for i in range(10):
    plt.subplot(ngibbs,10,g*10+i+1);  plt.axis('off')
    plt.imshow(flipped[i].reshape(28,28),'gray'); plt.draw()
    h = sig(1./T*(flipped[i] @ w+ hbias)) > np.random.rand(nhidden) 
    flipped[i] = sig(1./T*(w @ h + vbias)) > np.random.rand(nvisible)
