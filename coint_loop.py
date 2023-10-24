import pandas as pd
dates = pd.date_range("20220101", periods=5)
df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 
                   'B': [5, 6, 7, 8, 9], 
                   'C': [9, 8, 7, 6, 5]}, index=dates)

from statsmodels.tsa.stattools import adfuller

def test_cointegration(series1, series2):
    # Take the difference
    diff_series = series1 - series2

    # Perform ADF test
    result = adfuller(diff_series)

    # If p-value is less than 0.05, then we can reject the null hypothesis and consider the series to be cointegrated
    if result[1] < 0.05:
        return True
    else:
        return False

import numpy as np

assets = df.columns
n = len(assets)
matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i != j:
            matrix[i][j] = test_cointegration(df[assets[i]], df[assets[j]])
        else:
            matrix[i][j] = np.nan

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 7))
sns.heatmap(matrix, xticklabels=assets, yticklabels=assets, cmap='coolwarm', annot=True, cbar=False)
plt.title('Cointegration Matrix (1: Cointegrated, 0: Not Cointegrated)')
plt.show()




