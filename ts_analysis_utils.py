# Multi subplots

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Number of commodities
n_commodities = len(df_cleaned.columns)

# Create subplots
fig, axes = plt.subplots(nrows=n_commodities, figsize=(12, 15), sharex=True)

# Plot each series in a separate subplot
for i, column in enumerate(df_cleaned.columns):
    axes[i].plot(df_cleaned.index, df_cleaned[column], label=column, color='blue')
    axes[i].set_title(column)
    axes[i].grid(True)

# Adjust layout
plt.tight_layout()
plt.xlabel("Date")
plt.show()

# Timeseries decomposion / seasonal

# Filling missing values using forward fill
df_cleaned_filled = df_cleaned.fillna(method='ffill')

# Time Series Decomposition for 'Corn' with the filled values
decomposition = sm.tsa.seasonal_decompose(df_cleaned_filled['C'], model='additive')
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Plotting the decomposition
fig, axes = plt.subplots(4, 1, figsize=(12, 10))

# Original Series
axes[0].plot(df_silva_cleaned_filled['C'], label='Original', color='blue')
axes[0].set_title('Original Series')
axes[0].grid(True)

# Trend
axes[1].plot(trend, label='Trend', color='green')
axes[1].set_title('Trend')
axes[1].grid(True)

# Seasonality
axes[2].plot(seasonal, label='Seasonal', color='red')
axes[2].set_title('Seasonal')
axes[2].grid(True)

# Residuals
axes[3].plot(residual, label='Residual', color='purple')
axes[3].set_title('Residual')
axes[3].grid(True)

plt.tight_layout()
plt.show()

corn_stats, correlation_with_corn

# Another version of  plots with some risk metrics

# Sample data
dates = pd.date_range(start="2020-01-01", periods=365)
returns = np.random.randn(365) * 0.02  # Random daily returns
series = pd.Series(returns, index=dates)

def plot_cumulative_return(series):
    cumulative_return = (1 + series).cumprod() - 1
    cumulative_return.plot(ax=ax, title="Cumulative Return")
    ax.set_ylabel("Cumulative Return")

def plot_cumulative_dollar_value(series, initial_investment=1000):
    dollar_value = (1 + series).cumprod() * initial_investment
    dollar_value.plot(ax=ax, title="Cumulative Dollar Value of $1000 Investment")
    ax.set_ylabel("Dollar Value")

def plot_drawdown(series):
    cumulative_return = (1 + series).cumprod() - 1
    running_max = cumulative_return.cummax()
    drawdown = cumulative_return - running_max
    ax.fill_between(drawdown.index, drawdown, color="red", alpha=0.3)
    ax.set_title("Drawdown Plot")
    ax.set_ylabel("Drawdown")

def plot_timeseries_decomposition(series, model='additive'):
    decomposition = seasonal_decompose(series, model=model)
    trend = decomposition.trend.dropna()
    seasonal = decomposition.seasonal.dropna()
    residual = decomposition.resid.dropna()

    ax[0].plot(trend, label="Trend")
    ax[0].set_title("Trend")
    ax[1].plot(seasonal, label="Seasonality")
    ax[1].set_title("Seasonality")
    ax[2].plot(residual, label="Residual")
    ax[2].set_title("Residual")

fig, ax = plt.subplots(4, 1, figsize=(12, 15))

# Plotting
plot_cumulative_return(series)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(4, 1, figsize=(12, 15))
plot_cumulative_dollar_value(series)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(4, 1, figsize=(12, 15))
plot_drawdown(series)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(3, 1, figsize=(12, 15))
plot_timeseries_decomposition(series)
plt.tight_layout()
plt.show()

####### Seasonal plotting with heatmap

# Sample data
dates = pd.date_range(start="2010-01-01", periods=365*10)
prices = np.exp(np.random.randn(365*10) * 0.02).cumprod()  # Simulate price series
log_returns = np.log(prices / prices.shift(1))
series = pd.Series(log_returns, index=dates).dropna()

# Convert daily log returns to monthly returns
monthly_returns = series.resample('M').sum()

# Create a pivot table with years as rows and months as columns
pivot_table = monthly_returns.pivot_table(index=monthly_returns.index.year, 
                                          columns=monthly_returns.index.month, 
                                          values=0, 
                                          aggfunc='sum')

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, cmap='RdYlGn', center=0, fmt=".2%")
plt.title("Monthly Returns Heatmap")
plt.ylabel("Year")
plt.xlabel("Month")
plt.show()
