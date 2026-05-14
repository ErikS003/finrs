import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------
# 1. Load EUR/USD data from FRED
# -----------------------------------------------------

url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXUSEU"

df = pd.read_csv(url)

df.columns = ["date", "eur_usd"]
df["date"] = pd.to_datetime(df["date"])
# FRED sometimes uses "." for missing values apparently
df["eur_usd"] = pd.to_numeric(df["eur_usd"], errors="coerce")
df = df[(df["date"] >= "2015-01-01") & (df["date"] <= "2026-05-08")]
df = df.dropna()
df = df.set_index("date")
#  log exchange rate and log returns
df["log_eur_usd"] = np.log(df["eur_usd"])
df["log_return"] = df["log_eur_usd"].diff()
# Remove first missing return
df_returns = df.dropna()
#basic info
print(df.head())
print(df.tail())

print("\nNumber of observations:", len(df))
print("\nSummary statistics for EUR/USD level:")
print(df["eur_usd"].describe())

print("\nSummary statistics for log returns:")
print(df_returns["log_return"].describe())


plt.figure(figsize=(10, 5))
plt.plot(df.index, df["eur_usd"])
plt.title("Daily EUR/USD Exchange Rate")
plt.xlabel("Date")
plt.ylabel("U.S. dollars per euro")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(df.index, df["log_eur_usd"])
plt.title("Log EUR/USD Exchange Rate")
plt.xlabel("Date")
plt.ylabel("log(EUR/USD)")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(df_returns.index, df_returns["log_return"])
plt.title("Daily Log Returns of EUR/USD")
plt.xlabel("Date")
plt.ylabel("Log return")
plt.grid(True)
plt.show()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plot_acf(df["eur_usd"], lags=60)
plt.title("ACF of EUR/USD Exchange Rate Level")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.show()

plt.figure(figsize=(10, 5))
plot_acf(df["log_eur_usd"], lags=60)
plt.title("ACF of Log EUR/USD Exchange Rate")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.show()

plt.figure(figsize=(10, 5))
plot_acf(df_returns["log_return"], lags=60)
plt.title("ACF of Daily Log Returns")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.show()

from statsmodels.tsa.stattools import adfuller

def adf_test(series, name):
    result = adfuller(series.dropna(), autolag="AIC")

    print(f"\nADF test for {name}")
    print("-" * 50)
    print(f"ADF statistic: {result[0]:.6f}")
    print(f"p-value:       {result[1]:.6f}")
    print(f"Lags used:     {result[2]}")
    print(f"Observations:  {result[3]}")

    print("Critical values:")
    for key, value in result[4].items():
        print(f"  {key}: {value:.6f}")

    if result[1] < 0.05:
        print("Conclusion: Reject H0. The series appears stationary.")
    else:
        print("Conclusion: Fail to reject H0. The series appears non-stationary.")

adf_test(df["eur_usd"], "EUR/USD level")
adf_test(df["log_eur_usd"], "log EUR/USD level")
adf_test(df_returns["log_return"], "daily log returns")


from statsmodels.tsa.stattools import kpss

def kpss_test(series, name):
    result = kpss(series.dropna(), regression="c", nlags="auto")

    print(f"\nKPSS test for {name}")
    print("-" * 50)
    print(f"KPSS statistic: {result[0]:.6f}")
    print(f"p-value:        {result[1]:.6f}")
    print(f"Lags used:      {result[2]}")

    print("Critical values:")
    for key, value in result[3].items():
        print(f"  {key}: {value:.6f}")

    if result[1] < 0.05:
        print("Conclusion: Reject H0. The series appears non-stationary.")
    else:
        print("Conclusion: Fail to reject H0. The series appears stationary.")

kpss_test(df["eur_usd"], "EUR/USD level")
kpss_test(df["log_eur_usd"], "log EUR/USD level")
kpss_test(df_returns["log_return"], "daily log returns")