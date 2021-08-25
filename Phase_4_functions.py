import pandas as pd
import numpy as np
import itertools
import math
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler




def sarimax_search(train, p=range(1,3), d=range(1,2), q=range(1,3), maxiter=50,
                   enforce_stationarity=False, enforce_invertibility=False):
    
    """
    Takes in training data and p, q, d, and returns SARIMAX models with RMSE.  After all iterations,
    prints the best model and RMSE.
    
    Parameters
    ----------
    train = training data
    p: (range) for the iterations of p in SARIMAX model, default = range(1,3)
    d: (range) for the iterations of d in SARIMAX model, default = range(1,2)
    q: (range) for the iterations of q in SARIMAX model, default = range(1,3)
    maxiter: maximum interations, default=50
    enforce_stationarity:  default = False
    enforce_invertibility: default = False
    """
    
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]
    
    count = 0
    for param in pdq:
        for seasonal_param in seasonal_pdq:
            count += 1
    print(f'THERE ARE {count} ITERATIONS')
    answer = input("Would you like to run the models? (y or n)?")
    
    if (answer == 'y' or answer == 'yes'):
        lowest_RMSE = 1e10
        for param in pdq: 
            for seasonal_param in seasonal_pdq:
                try: 
                    model = SARIMAX(train, order=param, seasonal_order=seasonal_param,
                                   enforce_stationarity=enforce_stationarity, 
                                    enforce_invertibility=enforce_invertibility)
                    results = model.fit(maxiter=maxiter)
                    y_hat = results.predict(type='levels')
                    RMSE = np.sqrt(mean_squared_error(train, y_hat))
                    print(f'SARIMAX {param} x {seasonal_param} - RMSE:{RMSE}')
                    if RMSE < lowest_RMSE:
                        lowest_RMSE = RMSE
                        best_param = param
                        best_seasonal_param = seasonal_param
                except:
                    print('Oops!')
                    continue
        print(f'BEST RESULTS:  SARIMAX {best_param} x {best_seasonal_param} - RMSE:{lowest_RMSE}')  
    else:
        print('OK, SARIMAX models will not be run.')




def fbprophet_func(df, train_size=.8, periods=13):
    '''
    The input dataframe must only have two columns called ds and y

    Returns the Original and Predictions along with the RMSE of the test data

    Will split the data into a train, test for you.

    Height is for the height of the arrow

    periods is for the periods you wish to predict
    '''
    df_prophet = df
    cutoff = round(df_prophet.shape[0] * train_size)
    train = df_prophet[:cutoff]
    test = df_prophet[cutoff:]

    model = Prophet()
    model.fit(train)
    forecast = model.predict(train)

    future = model.make_future_dataframe(periods=periods, freq='MS')
    future_forecast = model.predict(future)

    fig, ax = plt.subplots()

    sns.lineplot(original['time'], original['value'],
                 label='Original', color='r', linewidth=4)
    sns.lineplot(
        future_forecast['ds'], future_forecast['yhat'], label='Predictions', color='b')
    model.plot(future_forecast, ax=ax)
    plt.vlines(x=train['ds'].max(), ymin=future_forecast['yhat'].min(
    ) - 5, ymax=future_forecast['yhat'].max() + 5, linestyles='dashed')
    ax.set_title('Denver House Prices')
    ax.set_ylabel('House Prices')
    ax.set_xlabel('Time')
    plt.legend(loc='best')

    MSE = np.square(np.subtract(test.y, future_forecast.yhat)).mean()

    RMSE = math.sqrt(MSE)
    print(f'RMSE for the test data: {RMSE}')
    print("%RMSE: ", RMSE / original.mean())




def naive_model(train, test, periods=1):
    
    """
    Takes in train and test data.  Shifts data by number of periods and returns a plot and RMSEs.
    
    Parameters
    ----------
    train: training data 
    test: test data 
    periods: number of periods to shift (default = 1)
    """
    
    naive_train = train.shift(periods=1)
    naive_test = test.shift(periods=1)
    
    fig = plt.figure(figsize=(15,8))
    plt.plot(train, label='Original Train')
    plt.plot(naive_train, label='Shifted Train')
    plt.plot(test, label='Original Test')
    plt.plot(naive_test, label='Shifted Test')
    plt.legend(loc='best')
    plt.title('Naive Model');
    
    RMSE_train = np.sqrt(mean_squared_error(train[1:], naive_train.dropna()))
    RMSE_test = np.sqrt(mean_squared_error(test[1:], naive_test.dropna()))
    
    print(f'The Naive Model RMSE for the train data is: {round(RMSE_train, 2)}')
    print(f'The Naive Model RMSE for the test data is: {round(RMSE_test, 2)}')

def DickeyFullerTest(ts):
    
    '''
    Takes in a time series and returns the results of the Dickey Fuller Test in a Panda Series format.
    '''

    dftest = adfuller(ts)

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '# Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' %key] = value

    return dfoutput

def decompose(ts):
    '''
    Takes in a time series and returns four plots: the orignal, trend, seasonal, and residuals.
    '''

    decomposition = seasonal_decompose(ts)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    plt.figure(figsize=(15,8))
    plt.subplot(411)
    plt.plot(ts, label='Original')
    plt.legend(loc='upper left')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='upper left')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonal')
    plt.legend(loc='upper left')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='upper left');

def roll_mean_std(ts, name='Enter Name'):
    """
    Takes a time series and name and returns a plot of rolling mean and std.
    
    Parameters
    ----------
    
    ts:  time series
    name:  ts name for the plot title.  (ex. '43222 (Columbus, OH)')
    """
    
    roll_mean = ts.rolling(window=12).mean()
    roll_std = ts.rolling(window=12).std()

    fig = plt.figure(figsize=(15,8))
    plt.plot(ts, label='Original')
    plt.plot(roll_mean, label='1 Yr. Rolling Mean')
    plt.plot(roll_std, label=('1 Yr. Rolling Std.'))
    plt.legend(loc='best')
    plt.title(f'{name} Rolling Mean & Std. Deviation');


def LSTM_func(df, City=None, verbose=1, use_multiprocessing=False, epochs=5, batch_size=5, test_size=0.8):
    '''
    The dataframe can only have two columns, time and values drop everthing else before using this function
    you will break it!!

    Parameters
    -----------
    df = Dataframe (DON'T FORGET ONLY TIME AND VALUES)
    City = The name of the city you are modeling

    Returns
    ---------
    A plot
    '''
    cols = list(df)[1:]

    train, test = df[:int(
        (len(df) * .80))], df[int(round(len(df) * test_size)):]

    train_float = train[cols].astype(float)
    test_float = test[cols].astype(float)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_float)
    test_scaled = scaler.transform(test_float)

    sequence_length = 5
    window_length = sequence_length + 1

    x_train = []
    y_train = []
    for i in range(0, len(train) - window_length + 1):
        window = train_scaled[i:i + window_length]
        x_train.append(window[:-1, :])
        y_train.append(window[-1, [-1]])
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_test = []
    y_test = []
    for i in range(0, len(test) - window_length + 1):
        window = test_scaled[i:i + window_length, :]
        x_test.append(window[:-1, :])
        y_test.append(window[-1, [-1]])
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # Model building starts here
    model = Sequential()
    model.add(LSTM(64, activation='relu',
                   input_shape=(5, 1), return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=True))
    model.add(LSTM(16, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.summary()
    # Fitting the model on the X_train and y_train
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(
        x_test, y_test), verbose=verbose, use_multiprocessing=use_multiprocessing)

    y_predicts = model.predict(x_train)
    y_predict_inv_scaled = scaler.inverse_transform(y_predicts)
    y_predict = []
    for predict in y_predict_inv_scaled:
        y_predict.append(predict[0])

    y_predict = pd.Series(y_predict, index=train.index[5:])

    y_test_inv_scaled = scaler.inverse_transform(y_train)
    y_train = []
    for predict in y_test_inv_scaled:
        y_train.append(predict[0])

    y_train = pd.Series(y_train, index=train.index[5:])
    
    MSE = np.square(np.subtract(y_train,y_predict)).mean() 
 
    RMSE = math.sqrt(MSE)
    print(f'RMSE: {RMSE}')
    print("%RMSE: ", RMSE/y_test.mean())


