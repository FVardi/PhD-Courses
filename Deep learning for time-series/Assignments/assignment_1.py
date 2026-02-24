# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
# %%

# Load data
df_raw = pd.read_csv("us_change.csv")

# %%
df = df_raw
df
# %%

x_name = "Income"
y_name = "y"

x = df[x_name]
y = df[y_name]

X = sm.add_constant(x)   # add intercept
ols_model = sm.OLS(y, X).fit()

beta = ols_model.params[x_name]
beta_se = ols_model.bse[x_name]
dw = sm.stats.stattools.durbin_watson(ols_model.resid)

print(f"β estimate    = {beta:.4f}")
print(f"Std. error    = {beta_se:.4f}")
print(f"Durbin–Watson = {dw:.3f}")


# %%
plt.figure()
plot_acf(ols_model.resid, lags=30)
plt.title("ACF of OLS residuals")
plt.show()


# %%

lb = acorr_ljungbox(ols_model.resid, lags=20, return_df=True)
print(lb)

# %%

resid = ols_model.resid

fig, ax = plt.subplots(2, 1, figsize=(10,6))

plot_acf(resid, lags=30, ax=ax[0])
ax[0].set_title("Residual ACF")

plot_pacf(resid, lags=30, ax=ax[1])
ax[1].set_title("Residual PACF")

plt.tight_layout()
plt.show()

# %%
import itertools

p_vals = range(0, 5)
q_vals = range(0, 5)

results = []

for p, q in itertools.product(p_vals, q_vals):
    try:
        model = SARIMAX(
            y,
            exog=x,
            order=(p, 0, q),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        res = model.fit(disp=False)

        results.append({
            "p": p,
            "q": q,
            "AIC": res.aic,
            "BIC": res.bic
        })

    except Exception:
        continue
# %%
results_df = pd.DataFrame(results).sort_values("BIC")
results_df
