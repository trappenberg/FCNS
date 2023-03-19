# PCA a la Oja
import numpy as np
import matplotlib.pyplot as plt

w = np.array([[-0.2,0.5]]).T; wTraj=w 
a = -np.pi/6 
rot = np.array([[np.cos(a),-np.sin(a)],
                [np.sin(a),np.cos(a)]]) 

# Training 
for i in range(1000):
   rPre = 0.05*np.random.normal([0,0],[1,4]); 
   rPre = rot@rPre
   plt.plot(rPre[0],rPre[1],'b.') 
   rPost = rPre@w 
   w = w+0.1*rPost*(rPre-rPost*w.T).T 
   wTraj = np.append(wTraj,w,axis=1) 
 
# Plotting results
plt.plot(wTraj[0,:],wTraj[1,:],'r')
plt.plot([0,w[0]],[0,w[1]],'k')
plt.plot([-1,1],[0,0],'k'); plt.plot([0,0],[-1,1],'k')
plt.axis([-1,1,-1,1])
