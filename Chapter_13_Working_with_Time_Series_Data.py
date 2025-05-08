
import pandas as pd

# Time series forecasting example (using a dataset like stock prices)
time_series_data = pd.read_csv('stock_data.csv', parse_dates=['Date'], index_col='Date')
time_series_data['Close'].plot(title='Stock Prices')
plt.show()
