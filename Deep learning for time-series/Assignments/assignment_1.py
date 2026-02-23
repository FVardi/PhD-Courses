# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
# %%

# Load data
y_raw = pd.read_csv("www_usage.csv")

# %%
df = y_raw.copy()

orders = (3, 1, 0)
model = ARIMA(df.y, order=orders)
model_fit = model.fit()
residuals = df.y - model_fit.predict()
residuals = residuals[1:]

dof = len(df) - sum(orders) + 1
lb_test = acorr_ljungbox(residuals)

plt.scatter(df.ds[1:], residuals)
plot_acf(residuals)

print(lb_test)
