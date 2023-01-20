#FitzHugh-Nagumo model
import matplotlib.pyplot as plt

# Parameters and initialization
Iext=0.5; v=[0]; w=[0]; t=[0]; dt=0.1
        
for step in range(1,int(200/dt)):
   # record time steps 
   t.append(t[step-1]+dt)
   # numerical integration and recording  
   v.append(v[step-1]+dt*(v[step-1]-v[step-1]**3/3-w[step-1]+Iext)) 
   w.append(w[step-1]+dt*0.08*(v[step-1]+0.7-0.8*w[step-1])) 
    
plt.plot(t,v); plt.xlabel('t'); plt.ylabel('v')