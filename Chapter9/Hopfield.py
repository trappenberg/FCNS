import numpy as np
import matplotlib.pyplot as plt
	
pat=2*np.random.randint(2,size=(500,10))-1#Rand binary pattern
w=pat@pat.T                               #Hebbian learning
s = pat+10*np.random.randn(500,10)        #Initialize network
for t in range(10):	s[:,t]=np.sign(w@s[:,t-1]) #Update network
plt.plot(s.T@pat/500)                     #plot overlaps
