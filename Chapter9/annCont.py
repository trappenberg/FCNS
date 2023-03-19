import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def udot(u, t, w, I_ext, dx, tau):
    r=1/(1+np.exp(-u)) 
    I_int=w@r*dx;
    return tau*(-u+I_int+I_ext)    

# Parameters 
nn = 400; dx = 1/nn; C = 0; tau = 1

# Training weight matrix with Hebbian learning and scaling
pat = 2*np.random.randint(2, size=(nn,10))-1
w = pat@pat.T; w = w/w[0,0]; w = 100*(w-C)

# Update with input (disturbed first pattern)
I_ext = pat[:,0]; I_ext[:10] = 1-I_ext[:10]
t = np.arange(10)
u0 = np.zeros(nn)  # Initial state
u = odeint(udot, u0, t, args=(w, I_ext, dx, tau))

# remove input 
I_ext = np.zeros(nn) 
t2 = np.arange(10,20)
u0 = u[-1,:]  # Initial state
u2 = odeint(udot, u0, t2, args=(w, I_ext, dx, tau))

# plot overlap
plt.plot(np.append(t,t2),np.append(u,u2,axis=0)@pat/nn)
plt.xlabel('time'); plt.ylabel('overlap')
