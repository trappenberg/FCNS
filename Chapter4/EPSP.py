# Synaptic conductance model to simulate an EPSP
import matplotlib.pyplot as plt

# Setting some constants and initial values
c_m=1; V_l=0; V_s=10; tau=1; dt=0.1;  
g_l=1; g_s=[0]; 
I_l=[0]; I_s=[0]; 
v_m=[0]; t=[0];

# Numerical integration using Euler scheme
for step in range(1,int(10/dt)):
    # record the time (in ms) in slot step of vector t 
    t.append(t[step-1]+dt)
    # simulate opening synaptic chanels around t=1ms  
    if abs(t[step]-1) < 0.0001: g_s[step-1] = 1;  
    # calculate the currents at this time 
    I_l.append(g_l * (v_m[step-1]-V_l)) 
    I_s.append(g_s[step-1] * (v_m[step-1]-V_s))  
    # update conductance and membrane potential 
    g_s.append(g_s[step-1]-dt/tau * g_s[step-1]) 
    v_m.append(v_m[step-1]-dt/c_m*(I_l[step]+I_s[step])) 
    
plt.plot(t,v_m,'k'); 