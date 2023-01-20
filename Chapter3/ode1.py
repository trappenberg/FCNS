# Example of numerical integration 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def xdot(x, t):
    return x

x0 = 1 # initial conditions on at x=0
dt = 0.1 # time step
t = np.arange(0,3,dt) # integrate from 0 to 4
 
xEuler = np.array([x0])
for step in t:
    xEuler = np.append(xEuler,
             xEuler[-1]+dt*xdot(xEuler[-1],step))

xODEint = np.squeeze(odeint(xdot, x0, t))

xAna = x0 * np.exp(t) # analytic solution

plt.plot(t,xEuler[:-1],'r--'); plt.plot(t,xAna) 
plt.xlabel('t'); plt.ylabel('xEuler, xAna')