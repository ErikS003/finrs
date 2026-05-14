from arch import arch_model
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
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
# Use daily log returns in percent
returns_pct = 100 * df_returns["log_return"].dropna()

# Fit GARCH(1,1) with constant mean
garch_model = arch_model(
    returns_pct,
    mean="Constant",
    vol="GARCH",
    p=1,
    q=1,
    dist="normal"
)

garch_fit = garch_model.fit(disp="off")

print(garch_fit.summary())
conditional_volatility = garch_fit.conditional_volatility

plt.figure(figsize=(10, 5))
plt.plot(conditional_volatility)
plt.title("Estimated Conditional Volatility from GARCH(1,1)")
plt.xlabel("Date")
plt.ylabel("Conditional volatility, percent")
plt.grid(True)
plt.show()

standardized_residuals = garch_fit.std_resid.dropna()

plt.figure(figsize=(10, 5))
plt.plot(standardized_residuals)
plt.title("Standardized Residuals from GARCH(1,1)")
plt.xlabel("Date")
plt.ylabel("Standardized residual")
plt.grid(True)
plt.show()

print("Ljung-Box test for standardized residuals:")
ljung_std = acorr_ljungbox(standardized_residuals, lags=[10, 20, 30], return_df=True)
print(ljung_std)

print("\nLjung-Box test for squared standardized residuals:")
ljung_std_sq = acorr_ljungbox(standardized_residuals ** 2, lags=[10, 20, 30], return_df=True)
print(ljung_std_sq)

garch_t_model = arch_model(
    returns_pct,
    mean="Constant",
    vol="GARCH",
    p=1,
    q=1,
    dist="t"
)

garch_t_fit = garch_t_model.fit(disp="off")

print(garch_t_fit.summary())

print("Normal GARCH AIC:", garch_fit.aic)
print("Student-t GARCH AIC:", garch_t_fit.aic)
print("Normal GARCH BIC:", garch_fit.bic)
print("Student-t GARCH BIC:", garch_t_fit.bic)