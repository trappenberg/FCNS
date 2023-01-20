# Wilson neuron
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def ydot(y,t,Iext,g,E,tau): 

  V=y[0]; R=y[1]; T=y[2]; H=y[3]
    
  gNa = 17.8 + 47.6*V + 33.8*V**2
  R0 = 1.24  +  3.7*V + 3.2*V**2
  T0 = 4.205 + 11.6*V + 8  *V**2
  X = np.array([R,gNa,T,H])
  
  Vdot = -1/tau[0]*(g*X@(V-E)-Iext)
  Rdot = -1/tau[1]*(R-R0)
  Tdot = -1/tau[2]*(T-T0)
  Hdot = -1/tau[3]*(H-3*T)
  
  return np.array([Vdot,Rdot,Tdot,Hdot])
  
# parameters of the model: 0=K  1=Na  2=T  3=H
g = np.array([26,1,2.25,9.5])
E = np.array([-0.95,0.50,1.20,-0.95])
tau = np.array([1,4.2,14,45])  
  
#1: Equilibration: no external input
Iext=0; y0=np.zeros(4); y0[0]=-1 
t=np.arange(0,100,0.1)
y = odeint(ydot, y0, t, args=(Iext,g,E,tau))
#2: Integration with external input
Iext=1; y0=y[-1,:];  
t=np.arange(0,200,0.1)
y = odeint(ydot, y0, t, args=(Iext,g,E,tau))
plt.plot(t,y[:,0]) 
plt.xlabel('Time'); plt.ylabel('Membrane potential')
