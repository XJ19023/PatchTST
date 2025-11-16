import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['axes.facecolor'] = 'gray'

x = np.arange(0, 10, 0.1)
y = np.sin(x)

# plt.plot(x, y, color='blue', label='sin(x)')
plt.fill_between(x, y, where=y>0, color='green', alpha=0.3)
plt.fill_between(x, y, where=y<0, color='red', alpha=0.3)

plt.savefig(f'aaa.png')