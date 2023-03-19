import numpy as np
import matplotlib.pyplot as plt
	
X=np.array([[0,0,1,1],
            [0,1,0,1],
            [1,1,1,1]])
Y=np.array([[0,1,1,1]])

# model specifications
Ni=3; No=1;
	
#parameter and array initialization
Ntrials=100
wo=np.random.randn(No,Ni); dwo=np.zeros(wo.shape) 
error1=np.array([])
error2=np.array([])
	
for trial in range(Ntrials):     
    y = 1/(1+np.exp(-wo@X)) #output for all pattern
    do = y*(1-y)*(Y-y)  # delta output
    # update weights with momentum
    dwo = 0.9*dwo+do@X.T
    wo = wo+0.5*dwo
    error1 = np.append(error1,np.sum((Y-y)**2))
    error2 = np.append(error2,np.sum(1-(abs(Y-y)<0.1)))

plt.plot(error1); plt.plot(error2/4)
