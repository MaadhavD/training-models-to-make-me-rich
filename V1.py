import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
from alpha_vantage.timeseries import TimeSeries
from tensorflow import * 
# Define Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = 'MGVOHERMHZ7BK0V7'

# Define function to get stock prices from Alpha Vantage
def get_stock_prices(symbol, interval='daily', outputsize='full'):
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    data, metadata = ts.get_daily_adjusted(symbol=symbol, outputsize=outputsize)
    data = data.iloc[::-1]
    data = data[['4. close']]
    data.columns = ['Close']
    return data

# Get stock prices for a symbol
symbol = 'AAPL'
prices = get_stock_prices(symbol)

# Define function to create the dataset for training the neural network
def create_dataset(data, lookback=60):
    X, y = [], []
    for i in range(len(data)-lookback-1):
        X.append(data[i:(i+lookback), 0])
        y.append(data[i+lookback, 0])
    return np.array(X), np.array(y)

# Create dataset
dataset = prices.values
lookback = 60
X, y = create_dataset(dataset, lookback)

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Define the neural network model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
early_stop = EarlyStopping(monitor='val_loss', patience=10)
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])

# Predict the stock prices for the test data
y_pred = model.predict(X_test)

# Visualize the results
import matplotlib.pyplot as plt
plt.plot(y_test, color='red', label='Actual')
plt.plot(y_pred, color='blue', label='Predicted')
plt.title('Stock Price Prediction for ' + symbol)
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
