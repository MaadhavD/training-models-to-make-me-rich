import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import datetime
import time

symbol = "AAPL"
start_date = '2020-03-31'
Key = "MGVOHERMHZ7BK0V7"
end_date = '2022-02-28'
start_timestamp = int(time.mktime(datetime.datetime.strptime(start_date, "%Y-%m-%d").timetuple()))
end_timestamp = int(time.mktime(datetime.datetime.strptime(end_date, "%Y-%m-%d").timetuple()))

print("running")


#print(end_timestamp)
#print(start_timestamp)
url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={Key}"

response = requests.get(url)
try:
    if response.status_code == 200:
        print(url)
        data = response.json()
        # load data into a dataframe
        df = pd.DataFrame(data).transpose()
        df = df[['open', 'high', 'low', 'close']]
        df = df.astype(float)  # convert data types to float
        print(df.colomns)
        X = df.drop('close', axis=1)
        y = df['close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        lin_reg = LinearRegression()
        lin_reg.fit(X, y)
        lin_reg_pred = lin_reg.predict(X)

        svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        svr.fit(X, y)
        svr_pred = svr.predict(X)

        mlp = MLPRegressor(random_state=1, max_iter=500)
        mlp.fit(X, y)
        mlp_pred = mlp.predict(X)

        bag_reg = BaggingRegressor(estimator=SVR(kernel='linear', C=100, gamma='scale'), n_estimators=10, random_state=0)
        bag_reg.fit(X, y)
        bag_reg_pred = bag_reg.predict(X)

        ada_reg = AdaBoostRegressor(estimator=LinearRegression(), n_estimators=100, random_state=0)
        ada_reg.fit(X, y)
        ada_reg_pred = ada_reg.predict(X)

        estimators = [('lr', LinearRegression()), ('svr', SVR(kernel='linear', C=100, gamma='scale')), ('mlp', MLPRegressor(random_state=1, max_iter=500))]
        stack_reg = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
        stack_reg.fit(X, y)
        stack_reg_pred = stack_reg.predict(X)

        all_pred = np.mean([lin_reg_pred, svr_pred, mlp_pred, bag_reg_pred, ada_reg_pred, stack_reg_pred], axis=0)

    #R-squared score
        r2 = r2_score(y, all_pred)
        print(f"R-squared score: {r2:.2f}")

        last_open = df.iloc[-1]['open']
        last_high = df.iloc[-1]['high']
        last_low = df.iloc[-1]['low']
        predicted_close = np.mean([
        lin_reg.predict([[last_open, last_high, last_low]]),
        svr.predict([[last_open, last_high, last_low]]),
        mlp.predict([[last_open, last_high, last_low]]),
        bag_reg.predict([[last_open, last_high, last_low]]),
        ada_reg.predict([[last_open, last_high, last_low]]),
        stack_reg.predict([[last_open, last_high, last_low]])
    ])
        print(f"Predicted closing price for next day: {predicted_close:.2f}")
        print("hello")
    else: 
        print(f"Error: {response.status_code}")

except Exception as e:
    print(e)
    print(df.columns)