# Hogdkin-Huxley model
import numpy as np
import matplotlib.pyplot as plt
# Max conductances (in units of mS/cm^2); 0=K,1=Na,2=R
g = np.array([36,120,0.3])
# Battery voltage ( in mV); 0=n, 1=m, 2=h
V = np.array([-12,115,10.613])
# Initialization of some variables
v=0; x = np.array([0,0,1]); t=0; dt=0.01
t_rec=[]; v_rec=[]; x_0_rec=[]
# Integration with Euler method
for step in range(1,int(100/dt)):
     t=t+dt
     if t>30 and t<89: I_ext=10 
     else: I_ext=0
  # alpha functions used by Hodgkin-and Huxley
     Alpha = np.array(
         [0.01*(-v+10)/(np.exp((-v+10)/10)-1),
          0.1*(-v+25)/(np.exp((-v+25)/10)-1),
          0.07*np.exp(-v/20)])
  # beta functions used by Hodgkin-and Huxley
     Beta = np.array([0.125*np.exp(-v/80),
                      4*np.exp(-v/18),
                      1/(np.exp((-v+30)/10)+1)])
  # update x = {n,m,h} variables
     x = x+dt*(Alpha*(1-x)-Beta*x)
  # calculate actual conductances g with given n, m, h
     gx=np.array([g[0]*x[0]**4,g[1]*x[1]**3*x[2],g[2]])
  # Ohm's law
     I = gx*(v-V);
  # update voltage of membrane
     v = v+dt*(I_ext-sum(I))
  # record some variables for plotting after equil.
     t_rec.append(t)
     v_rec.append(v)
     
# Plotting results
plt.plot(t_rec,v_rec); 
plt.xlabel('Time'); plt.ylabel('Voltage')
plt.rcParams.update({'font.size': 15})
plt.legend(loc="center left")