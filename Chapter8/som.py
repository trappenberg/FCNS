# Two dimensional self-organizing feature map al la Kohonen
import numpy as np
import matplotlib.pyplot as plt

nn = 10; lr = 0.02; sig = 2; sig2 = 1/(2*sig**2); ntrial = 0

x, y = np.meshgrid(np.arange(0, nn), np.arange(0, nn))

# Initial centres of prefered features:
c1=np.random.rand(nn,nn)
c2=np.random.rand(nn,nn)
#c1=0.05+x/nn
#c2=0.05+y/nn
# training session
for trial in range(10000):
    if (np.mod(trial,100)==0): # Plot grid of feature centres
        plt.axis([0,1,0,1]);
        plt.plot(c1,c2,'k'); plt.plot(c1.T,c2.T,'k'); 
        plt.show()
        
    r_in=[np.random.rand(),np.random.rand()]
    r=np.exp(-(c1-r_in[0])**2-(c2-r_in[1])**2)
    winner = np.unravel_index(np.argmax(r, axis=None), r.shape)
    r=np.exp(-((x-winner[0])**2+(y-winner[1])**2)*sig2);
    c1=c1+lr*r*(r_in[1]-c1)
    c2=c2+lr*r*(r_in[0]-c2)
