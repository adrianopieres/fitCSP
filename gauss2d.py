import numpy as np
import matplotlib.pyplot as plt

mean = (10, -1.5)
cov = [[1, 0], [0, 10]]  # diagonal covariance
x, y = np.random.multivariate_normal(mean, cov, 5000).T
plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()
