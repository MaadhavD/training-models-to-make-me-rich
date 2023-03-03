import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
from alpha_vantage.timeseries import TimeSeries

# Define Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = 'MGVOHERMHZ7BK0V7'

# Get stock prices for AAPL
def get_stock_prices(symbol, interval='daily', outputsize='full'):
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    data, metadata = ts.get_daily_adjusted(symbol=symbol, outputsize=outputsize)
    data = data.iloc[::-1]
    data = data[['4. close']]
    data.columns = ['Close']
    return data

# Create dataset
def create_dataset(data, lookback=60):
    X, y = [], []
    for i in range(len(data)-lookback-1):
        X.append(data[i:(i+lookback), 0])
        y.append(data[i+lookback, 0])
    return np.array(X), np.array(y)

# Get stock prices for AAPL
symbol = 'AAPL'
prices = get_stock_prices(symbol)

# Convert time values to dates
prices.index = pd.to_datetime(prices.index)

# Create dataset
dataset = prices.values
lookback = 60
X, y = create_dataset(dataset, lookback)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train LSTM model
early_stop = EarlyStopping(monitor='val_loss', patience=10)
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])

# Make predictions for future stock prices
future_prices = []
last_data = X[-1]
for i in range(12):
    prediction = model.predict(last_data.reshape(1, lookback, 1))
    future_prices.append(prediction[0][0])
    last_data = np.append(last_data[1:], prediction[0])

# Convert time values for future dates
future_dates = pd.date_range(start=prices.index[-1], periods=12, freq='D')

# Plot actual and predicted stock prices
import matplotlib.pyplot as plt
plt.plot(prices.index[-100:], prices['Close'].tail(100), color='red', label='Actual')
plt.plot(future_dates, future_prices, color='blue', label='Predicted')
plt.title('Stock Price Prediction for ' + symbol)
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
