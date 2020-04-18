import matplotlib.pyplot as plt
import numpy as np

N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2
plt.title('test plot')
plt.scatter(x, y, s = area, c=colors, alpha=0.5)
plt.show()

