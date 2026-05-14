import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

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

# df contains: eur_usd, log_eur_usd, log_return

train_end = "2024-12-31"

train = df[df.index <= train_end].copy()
test = df[df.index > train_end].copy()

train_returns = train["log_return"].dropna()
test_returns = test["log_return"].dropna()

print("Training observations:", len(train))
print("Test observations:", len(test))
print("Training return observations:", len(train_returns))
print("Test return observations:", len(test_returns))

# Random walk one step ahead forecast for the test period
test["rw_forecast"] = df["eur_usd"].shift(1).loc[test.index]

# Remove first missing forecast if needed
rw_eval = test.dropna(subset=["rw_forecast"])

# Forecast errors
rw_errors = rw_eval["eur_usd"] - rw_eval["rw_forecast"]

rw_mae = np.mean(np.abs(rw_errors))
rw_rmse = np.sqrt(np.mean(rw_errors ** 2))

print("Random walk forecast performance:")
print(f"MAE:  {rw_mae:.6f}")
print(f"RMSE: {rw_rmse:.6f}")

plt.figure(figsize=(10, 5))
plt.plot(rw_eval.index, rw_eval["eur_usd"], label="Actual EUR/USD")
plt.plot(rw_eval.index, rw_eval["rw_forecast"], label="Random walk forecast")
plt.title("Random Walk Forecast of EUR/USD Level")
plt.xlabel("Date")
plt.ylabel("EUR/USD")
plt.legend()
plt.grid(True)
plt.show()

from statsmodels.tsa.arima.model import ARIMA

# Fit ARMA(0,0) to training returns
arma_model = ARIMA(train_returns, order=(0, 0, 0))
arma_fit = arma_model.fit()

mu_hat = arma_fit.params["const"]
print("Estimated mean return:", mu_hat)

# Forecast returns over the test period
arma_return_forecast = pd.Series(mu_hat, index=test_returns.index)

# Return forecast errors
arma_errors = test_returns - arma_return_forecast

arma_mae = np.mean(np.abs(arma_errors))
arma_rmse = np.sqrt(np.mean(arma_errors ** 2))

print("ARMA(0,0) return forecast performance:")
print(f"MAE:  {arma_mae:.6f}")
print(f"RMSE: {arma_rmse:.6f}")

plt.figure(figsize=(10, 5))
plt.plot(test_returns.index, test_returns, label="Actual log returns")
plt.plot(arma_return_forecast.index, arma_return_forecast, label="ARMA(0,0) forecast")
plt.title("ARMA(0,0) Forecast of Daily Log Returns")
plt.xlabel("Date")
plt.ylabel("Log return")
plt.legend()
plt.grid(True)
plt.show()

from arch import arch_model

# Scale returns to percent
train_returns_pct = 100 * train_returns
test_returns_pct = 100 * test_returns

# Fit Student-t GARCH(1,1)
garch_model = arch_model(
    train_returns_pct,
    mean="Constant",
    vol="GARCH",
    p=1,
    q=1,
    dist="t"
)

garch_fit = garch_model.fit(disp="off")
print(garch_fit.summary())

rolling_vol_forecasts = []

combined_returns_pct = 100 * df["log_return"].dropna()

test_index = test_returns.index

for date in test_index:
    train_sample = combined_returns_pct[combined_returns_pct.index < date]

    model = arch_model(
        train_sample,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="t"
    )

    fit = model.fit(disp="off")

    forecast = fit.forecast(horizon=1, reindex=False)
    variance_forecast = forecast.variance.values[-1, 0]
    volatility_forecast = np.sqrt(variance_forecast)

    rolling_vol_forecasts.append(volatility_forecast)

rolling_garch_vol = pd.Series(rolling_vol_forecasts, index=test_index)

realized_abs_returns = np.abs(100 * test_returns)

plt.figure(figsize=(10, 5))
plt.plot(realized_abs_returns.index, realized_abs_returns, label="Absolute return")
plt.plot(rolling_garch_vol.index, rolling_garch_vol, label="Rolling GARCH volatility forecast")
plt.title("Rolling One-Step-Ahead GARCH Volatility Forecast")
plt.xlabel("Date")
plt.ylabel("Percent")
plt.legend()
plt.grid(True)
plt.show()

realized_variance_proxy = (100 * test_returns) ** 2
forecast_variance = rolling_garch_vol ** 2

vol_errors = realized_variance_proxy - forecast_variance

garch_var_mae = np.mean(np.abs(vol_errors))
garch_var_rmse = np.sqrt(np.mean(vol_errors ** 2))

print("GARCH variance forecast performance:")
print(f"MAE:  {garch_var_mae:.6f}")
print(f"RMSE: {garch_var_rmse:.6f}")

vol_corr = np.corrcoef(rolling_garch_vol, realized_abs_returns)[0, 1]
print(f"Correlation between forecasted volatility and absolute returns: {vol_corr:.4f}")

actual_direction = np.sign(test_returns)
forecast_direction = np.sign(arma_return_forecast)

directional_accuracy = np.mean(actual_direction == forecast_direction)

print(f"Directional accuracy: {directional_accuracy:.4f}")