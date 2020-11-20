import keras

from alpha_vantage.techindicators import TechIndicators
import matplotlib.pyplot as plt
from click._compat import raw_input
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np

from yahoofinancials import YahooFinancials as yf
import pendulum
import datetime
import holidays
import os

def get_stock_data(symbol='^VIX',start_date='2020-01-01'):
    today = str(pendulum.today().date())
    stock = yf(symbol)
    prices = stock.get_historical_price_data(start_date,today,'daily')
    prices = pd.json_normalize(prices)
    prices = pd.json_normalize(prices[('%s.prices' % (symbol))][0])
    return prices

class TechnicalIndicators:
    def __init__(self,symbol='AAPL',request_input=True,get_technicals = False):
        self.api_key= 'KMOA5YH5JT2Z2A1B'
        if request_input:
            self.stock_name=self.question()
        else:
            self.stock_name=symbol
            
        if get_technicals:
            self.macd_data=self.macd()
            self.rsi_data=self.rsi()
            self.bbands_data=self.bbands()
            self.sma_data=self.sma()
        
        self.close_data=self.close()      
        self.vix_data = self.vix()
        
    def question(self):
        stock_name=raw_input("Enter stock name:")
        return stock_name
    def macd(self):
        a = TechIndicators(key=self.api_key, output_format='pandas')
        data, meta_data = a.get_macd(symbol=self.stock_name,interval='daily')
        return data
    def rsi(self):
        b=TechIndicators(key=self.api_key,output_format='pandas')
        data,meta_data = b.get_rsi(symbol=self.stock_name,interval='daily',time_period=14)
        return data
    def bbands (self):
        c=TechIndicators(key=self.api_key,output_format='pandas')
        data,meta_data=c.get_bbands(symbol=self.stock_name)
        return data
    def sma(self):
        d= TechIndicators(key=self.api_key, output_format='pandas')
        data, meta_data = d.get_sma(symbol=self.stock_name,time_period=30)
        return data
    def close(self):
        #d=TimeSeries(key=self.api_key,output_format='pandas')
        #data,meta_data=d.get_daily(symbol=self.stock_name,outputsize='full')
        data = get_stock_data(symbol=self.stock_name,start_date='1999-12-17')
        data['date'] = pd.to_datetime(data['formatted_date'], format='%Y-%m-%d')
        data = data.set_index('date')
        data = data[['close','volume']]
        return data
    def vix(self):
        data = get_stock_data(symbol='^VIX',start_date='1999-12-17')
        data['date'] = pd.to_datetime(data['formatted_date'], format='%Y-%m-%d')
        data = data.set_index('date')
        data = data[['close']]
        data.columns = ['VIX']
        return data
    


def next_business_day(from_date):
    ONE_DAY = datetime.timedelta(days=1)
    HOLIDAYS_US = holidays.US()

    next_day = from_date + ONE_DAY
    while next_day.weekday() in holidays.WEEKEND or next_day in HOLIDAYS_US:
        next_day += ONE_DAY
    return next_day

def create_prediction_array(X, scaler, lag = 60,forward_range = 5):

    values_x = X.values

    inputs_x = values_x[(values_x.shape[0] - lag):]
    inputs_x = scaler.transform(inputs_x)

    inputs_x = inputs_x.reshape(inputs_x.shape[0],1,inputs_x.shape[1])

    x_test_lag = []
    for x in range(lag,inputs_x.shape[0]+1):
        x_test_lag.append(inputs_x[(x-lag):x,0])

    x_test_lag = np.array(x_test_lag)
    x_test_lag = np.reshape(x_test_lag,(x_test_lag.shape[0],x_test_lag.shape[1],X.shape[1]))

    return x_test_lag


def getdata(data,window_size):
    X,y = np.array([1]*window_size),np.array([])
    for i in range(window_size, len(data)):
        X = np.vstack((X,data[i-window_size:i]))
        y = np.append(y,data[i:i+1])

    return X[1:],y

def latest_predictions(symbol, root_dir='Stock-Prediction-models', use_alpha_vantage = False):

    lags = [7]#[3,3,7] 
    forward_looking_days_con = [1]#[5,3,1]
    
    TI=TechnicalIndicators(symbol=symbol, request_input=False,get_technicals = use_alpha_vantage)
    
    if use_alpha_vantage:
        macd_data=TI.macd_data
        rsi_data=TI.rsi_data
        bbands_data=TI.bbands_data
        sma_data=TI.sma_data
    
    vix_data = TI.vix_data
    close_data=TI.close_data[['close']]
    volume_data=TI.close_data[['volume']]
    
    predictions = {}

    for i in range(len(lags)):
        print('Starting process for: %s-days lag and %s-days forward prediction' % (lags[i],forward_looking_days_con[i]))

        lag, forward_looking_days = lags[i], forward_looking_days_con[i]

        if forward_looking_days == 1:
            window_size = 7
            dataset = close_data['close'].values[-150:]
            predictions_custom = close_data[-100:].copy()
            predictions_custom.columns = ['Actual Price']

            # Calling `save('my_model')` creates a SavedModel folder `my_model`.
            path_model = os.path.dirname(__file__) +'/%s/%s_model_gr_%sday_lag_%sday_forward' % (root_dir, symbol, lag, forward_looking_days)
            reconstructed_model = eval("keras.models.load_model(path_model)")

            X_,y_ = getdata(close_data['close'].values[-150:],window_size)
            yhat = reconstructed_model.predict(X_[-100:])

            predictions_custom['Predicted Price'] = yhat

            X = np.array([1]*window_size)
            for i in range(window_size, len(dataset)+1):
                X = np.vstack((X,dataset[i-window_size:i]))

            X = X[1:]
            yhat_new = reconstructed_model.predict(X[-1:])

            next_bus_day = next_business_day(close_data.index[-1])
            predictions_custom.loc[next_bus_day, 'Predicted Price'] = yhat_new[0][0]

            predictions[forward_looking_days] = predictions_custom

        else:
            if forward_looking_days == 3:
                dataset = pd.concat([volume_data.copy(), vix_data.copy(), close_data.copy()], axis=1,sort=True).reindex(macd_data.index)
            else:
                dataset = pd.concat([rsi_data.copy(), volume_data.copy(), vix_data.copy(), close_data.copy()], axis=1,sort=True).reindex(macd_data.index)

            dataset = dataset.sort_index(ascending=True)

            # Load the x-scaler
            scaler_x = eval("load(open('%s/%s_scaler_x_%sdays_forward.pkl', 'rb'))" % (root_dir, symbol, forward_looking_days))

            x_test_lag = create_prediction_array(X = dataset[-100:], scaler = scaler_x, lag = lag, forward_range = forward_looking_days)


            # Calling `save('my_model')` creates a SavedModel folder `my_model`.
            reconstructed_model = eval("keras.models.load_model('%s/%s_model_gr_%sday_lag_%sday_forward')" % (root_dir, symbol, lag, forward_looking_days))

            # Load the y-scaler
            scaler_y = eval("load(open('%s/%s_scaler_y_%sdays_forward.pkl', 'rb'))" % (root_dir, symbol, forward_looking_days))


            yhat = reconstructed_model.predict(x_test_lag)

            inv_yhat = scaler_y.inverse_transform(yhat)

            predictions[forward_looking_days] = (float(inv_yhat),dataset['close'][-1])

    return predictions

        
if __name__ == "__main__":
    TI=TechnicalIndicators()
    close_data = TI.close_data
    macd_data = TI.macd_data
    rsi_data=TI.rsi_data
    bbands_data=TI.bbands_data
    sma_data = TI.sma_data
    plt.plot(macd_data)
    plt.show()
