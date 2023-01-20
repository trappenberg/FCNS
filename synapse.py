# voltage over time for non-NMDA synapse
import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0,10,0.1)
v = t*np.exp(-t/2)
plt.plot(t,v); plt.xlabel("t"); plt.ylabel("v")
