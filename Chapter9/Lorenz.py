# Plot trajectory of Lorenz system
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def udot(t,u):
    a=10.; b=28.; c=8/3.
    return a*(u[1] - u[0]), u[0]*(b-u[2])-u[1], u[0]*u[1]-c*u[2] 

u0 = [1.0, 1.0, 1.0]  # Initial state
t_span = (0.0, 30.0)
t = np.arange(0.0, 30.0, 0.01)
u = solve_ivp(udot, t_span, u0, method='LSODA', t_eval=t)
fig = plt.figure(); ax = plt.axes(projection='3d')
ax.plot(u.y[0, :], u.y[1, :], u.y[2, :])
