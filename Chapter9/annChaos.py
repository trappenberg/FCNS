import numpy as np
import matplotlib.pyplot as plt

# Assymetric network model
nn = 251; dt = 0.1

wa = np.random.normal(size=(nn,nn)) 
ws = np.random.normal(size=(nn,nn))
for i in range(nn):
    wa[i,i] = 0; ws[i,i] = 0
    for j in range(i):
        wa[i,j] = -wa[j,i]
        ws[i,j] = ws[j,i]

u_dif = np.zeros((41,41))
for a in range(41): 
    print(a)
    gs = (a-21)*0.5
    for b in range(41):
        ga = (b-21)*.5
        w = gs*ws+ga*wa
        u = 2*np.random.random(size=(nn,1))-1
        for t in np.arange(0,50,dt):
            s = np.tanh(u); u = (1-dt)*u+dt*w@s
        norm1 = u.T@u
        for t in np.arange(0,10,dt):
            s = np.tanh(u); u = (1-dt)*u+dt*w@s            
        u_dif[a,b]=np.sqrt(abs(u.T@u-norm1))

plt.imshow(u_dif)
