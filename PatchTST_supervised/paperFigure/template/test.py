import numpy as np
import matplotlib.pyplot as plt
# Create some fake data.
x1 = np.linspace(0.0, 5.0)
y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
x2 = np.linspace(0.0, 2.0)
y2 = np.cos(2 * np.pi * x2)


fig, axes = plt.subplots(
    2, 1, figsize=(8, 6), dpi=300,
    gridspec_kw={'height_ratios': [1, 1]}  # 上下图比例1：1
)
plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('A tale of 2 subplots')
plt.ylabel('Damped oscillation')

plt.subplot(2, 1, 2)
plt.plot(x2, y2, '.-')
plt.xlabel('time (s)')
plt.ylabel('Undamped')

plt.savefig(f'aaa.png')