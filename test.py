
# %%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# %%

X = np.array([[-1], [1], [2]])
y = np.array([[1], [1.5], [3]])

reg = LinearRegression().fit(X, y)


# %%

t = np.linspace(-1, 2)
y_fit = reg.coef_*t + reg.intercept_

plt.scatter(X, y)
plt.plot(t.reshape(-1, 1), y_fit.reshape(-1, 1))