import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy.integrate import odeint

def udot(u, t, w, I_ext, dx, tau):
    r=1/(1+np.exp(-u)) 
    I_int=w@r*dx;
    return tau*(-u+I_int+I_ext)    

# Parameters
nn = 100; dx = 2*np.pi/nn; sig = 2*np.pi/10; C=0.5; tau=1

# Training weight matrix with Hebbian learning and scaling
i = np.arange(nn)
pat = np.zeros((nn,nn))
for loc in range(nn):
    dis = np.fmin(abs(i-loc),nn-abs(i-loc))
    pat[:,loc] = np.exp(-(dis*dx)**2/(2*sig**2))
w = pat@pat.T; w = w/w[0,0]; w = 4*(w-C)

# Update with input (central activation) 
I_ext = np.zeros(nn); I_ext[int(nn/2-nn/10):int(nn/2+nn/10)] = 1
t = np.arange(10)
u0 = np.zeros(nn)  # Initial state
u = odeint(udot, u0, t, args=(w, I_ext, dx, tau))

# remove input 
I_ext = np.zeros(nn) 
t2 = np.arange(10,20)
u0 = u[-1,:]  # Initial state
u2 = odeint(udot, u0, t2, args=(w, I_ext, dx, tau))


# plot results
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
X = np.append(t,t2); Z = np.append(u,u2,axis=0).T
Y = np.arange(nn)/nn*2*np.pi
X, Y = np.meshgrid(X, Y)
ax.plot_surface(X,Y,Z,cmap=cm.coolwarm,linewidth=0,antialiased=False)
