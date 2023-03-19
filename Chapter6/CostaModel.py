# Costa model of synaptic dynamics
import numpy as np
import matplotlib.pyplot as plt

tmax=500; dt=0.1
X = np.zeros(tmax*int(1/dt)); a=[500,1000,1500,3000,3500,4000]; X[a]=1
Y = np.zeros(tmax*int(1/dt)); b=[600,1100,1600,2900,3400,3900]; Y[b]=1

tauym = 32.7; tauyp = 230.2; tauxp = 66.6; D = 200; F = 50
cp = 0.0618; dp = 0.1548; dm = 0.1771
xp = np.array([0]); yp = np.array([0]); ym = np.array([0])
P = np.array([0.7]); q = np.array([0.7])
r = np.array([1]); p = np.array([P])

for t in range(0,tmax*int(1/dt)):
    
    xp = np.append(xp,xp[t]+dt*(-1/tauxp*xp[t])+X[t])
    yp = np.append(yp,yp[t]+dt*(-1/tauyp*yp[t])+Y[t])
    ym = np.append(ym,ym[t]+dt*(-1/tauym*ym[t])+Y[t])

    r = np.append(r,r[t]+dt*(1-r[t])/D - dt*X[t]*p[t]*r[t])
    p = np.append(p,p[t]+dt*(P[t]-p[t])/F + X[t]*P[t]*(1-p[t]))
    
    q = np.append(q,q[t]+Y[t]*cp*xp[t]*ym[t-1])
    P = np.append(P,P[t]+X[t]*(-dm*ym[t]+dp*xp[t-1])*yp[t])
    if P[-1]<0: P[-1]=0 
    if P[-1]>1: P[-1]=1 
    if q[-1]<0: q[-1]=0 
    if q[-1]>2: q[-1]=2

plt.plot(P*q)
