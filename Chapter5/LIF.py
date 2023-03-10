# Simulation of (leaky) integrate-and-fire neuron
import numpy as np
import matplotlib.pyplot as plt

# parameters of the model
dt=0.1       # integration time step [ms]
tau=10       # time constant [ms]
E_L=-65      # resting potential [mV]
theta=-55    # firing threshold [mV]
RI_ext=8   # constant external input [mA/Ohm]
 
# Integration with Euler method
v_rec=np.array([])
t_rec=np.array([])
s_rec=np.array([])
t_step=0; v=E_L
for t in range(int(100/dt)):
    s=v>theta
    v=s*E_L+(1-s)*(v-dt/tau*((v-E_L)-RI_ext))
    v_rec=np.append(v_rec,v)
    t_rec=np.append(t_rec,t)
    s_rec=np.append(s_rec,s)

# Plotting results
ax1 = plt.axes([0.2, 0.7, 0.7, 0.2])
ax1.plot(t_rec,s_rec,'.',markersize=20)
ax1.axis([0, 100/dt, 0.5, 1.5])
plt.xticks([], []); plt.yticks([], [])
plt.ylabel('Spikes')

ax2 = plt.axes([0.2, 0.2, 0.7, 0.5])
ax2.plot(t_rec,v_rec)
ax2.plot([0, 100/dt],[-55, -55],'--');
ax2.axis([0, 100/dt, -66, -53])
plt.xlabel('Time [ms]'); plt.ylabel('v [mV]')

