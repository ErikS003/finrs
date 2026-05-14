from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXUSEU"

df = pd.read_csv(url)

df.columns = ["date", "eur_usd"]
df["date"] = pd.to_datetime(df["date"])
df["eur_usd"] = pd.to_numeric(df["eur_usd"], errors="coerce")
df = df[(df["date"] >= "2015-01-01") & (df["date"] <= "2026-05-08")]
df = df.dropna()
df = df.set_index("date")
df["log_eur_usd"] = np.log(df["eur_usd"])
df["log_return"] = df["log_eur_usd"].diff()
df_returns = df.dropna()
returns = df_returns["log_return"].reset_index(drop=True)
orders = [
    (0, 0, 0),
    (1, 0, 0),
    (0, 0, 1),
    (1, 0, 1),
    (2, 0, 0),
    (0, 0, 2),
    (2, 0, 1),
    (1, 0, 2),
    (2, 0, 2),
    (3, 0, 0),
    (0, 0, 3),
    (3, 0, 1),
    (1, 0, 3)
]

results = []

for order in orders:
    try:
        model = ARIMA(returns, order=order)
        fitted = model.fit()

        results.append({
            "Model": f"ARMA({order[0]},{order[2]})",
            "p": order[0],
            "q": order[2],
            "AIC": fitted.aic,
            "BIC": fitted.bic,
            "LogLik": fitted.llf
        })

    except Exception as e:
        print(f"Model ARMA({order[0]},{order[2]}) failed: {e}")

model_comparison = pd.DataFrame(results)
model_comparison = model_comparison.sort_values("BIC")

print(model_comparison)
print(model_comparison.sort_values("AIC"))
chosen_order = (0, 0, 0)  

chosen_model = ARIMA(returns, order=chosen_order)
chosen_fit = chosen_model.fit()

print(chosen_fit.summary())

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

residuals = chosen_fit.resid

plt.figure(figsize=(10, 5))
plt.plot(residuals)
plt.title("Residuals from ARMA(0,0) Model")
plt.xlabel("Time")
plt.ylabel("Residual")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plot_acf(residuals, lags=60)
plt.title("ACF of ARMA(0,0) Residuals")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.show()

print("Ljung-Box test for residuals:")
ljung_resid = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
print(ljung_resid)

squared_residuals = residuals ** 2

plt.figure(figsize=(10, 5))
plot_acf(squared_residuals, lags=60)
plt.title("ACF of Squared ARMA(0,0) Residuals")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.show()

print("Ljung-Box test for squared residuals:")
ljung_squared = acorr_ljungbox(squared_residuals, lags=[10, 20, 30], return_df=True)
print(ljung_squared)