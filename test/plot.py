import matplotlib.pyplot as plt
import numpy as np
from math import log

c = 0.05
x = np.linspace(0, 100, 1000)
#y = - 1/np.exp(c*x) + 1
#y = x/100

k = 10
y = np.exp(-(x - k)**2/100**2) - np.exp(-k**2/100**2)

print(x[0], y[-1])
plt.plot(x, y)
plt.show()
