# references 
# https://otexts.com/fpp2/holt-winters.html 
# https://medium.com/better-programming/exponential-smoothing-methods-for-time-series-forecasting-d571005cdf80

import pandas as pd
import numpy as np
import time

import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing # https://www.statsmodels.org/dev/generated/statsmodels.tsa.holtwinters.ExponentialSmoothing.html
from sklearn.metrics import mean_squared_error, r2_score

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', RuntimeWarning)

# plt.rcParams["figure.figsize"] = (20,10)

def ExponencialSmothingModel(target_pollutant, data_type='day', ahead=1):
    data = pd.read_csv('madrid_air_quaility_2001-2018.csv', sep=',', header=0,  index_col='date', parse_dates=True)
    station_data = pd.read_csv('stations.csv', sep=',', header=0, index_col=0)
    valid_stations_list = station_data.index.to_list()
    data = data[data.station.isin(valid_stations_list)]

    if data_type == 'day':
        data = data[target_pollutant].dropna().resample('D').mean()
        seasonal_type = 'multiplicative'
        seasonal_period = 365
    elif data_type == 'month':
        data = data[target_pollutant].dropna().resample('D').mean().resample('M').max()
        seasonal_type = 'additive'
        seasonal_period = 12

    data = data.fillna(method='ffill') # propagate last valid observation forward to next valid observation

    train = data[data.index <  '2016']
    test  = data[data.index >= '2016']
    
    plt.style.use('bmh')
    one_ahead_pred = []
    two_ahead_pred = []
    print('Predicting pollutant ' + target_pollutant + ' for ' + str(ahead) + ' ' + data_type + ' ahead')
    t0 = time.time()
    test = test[0:30] # test for just 30 days just to test quicker
    for t in test:
        model = ExponentialSmoothing(train, seasonal='add', seasonal_periods=seasonal_period).fit()
        prediction = model.forecast(2)
        prediction
        one_ahead_pred.append(prediction[0])
        if ahead>1:
            two_ahead_pred.append(prediction[1])
        train = np.append(train, t)
    t1 = time.time()
    training_time = t1-t0
    t_sec = str(training_time%60)
    t_min = str(int(training_time/60))
    print(t_min +' min and '+t_sec+' seconds')

    def evaluation(prediction, test, filename):
        mse = mean_squared_error(test.values, prediction)
        r2  = r2_score(test.values, prediction)
        error = abs(test.values - prediction)
        print('R2', round(r2,4))
        print('MSE', round(mse,4))
        print('MAE', round(np.array(error).mean(),4))

        pollutant_color = {'SO_2':'limegreen', 'O_3':'dodgerblue', 'NO_2':'magenta', 'PM10':'blueviolet'}
        plt.figure(filename+'ahead')
        plt.plot(test.index, test, color='lightcoral', label='Test')
        plt.plot(test.index, prediction,  color=pollutant_color.get(target_pollutant) ,label='Prediction')
        plt.legend(loc='best')
        plt.scatter(test.index, test, color='lightcoral', label='Test')
        plt.scatter(test.index, prediction,  color=pollutant_color.get(target_pollutant) ,label='Prediction')
        plt.ylabel(target_pollutant+' (\u03BCg/m\u00b3)')
        # plt.savefig('figures/esm/'+filename)
        # plt.close()
        plt.show()
    print('Model evaluation 1 ' + data_type + ' ahead')
    evaluation(one_ahead_pred, test, filename=target_pollutant+data_type+'1')
    if ahead>1:
        print('Model evaluation '+str(ahead)+' ' +data_type + ' ahead')
        evaluation(two_ahead_pred[:-1], test[(ahead-1):],  filename=target_pollutant+data_type+'2')


primary_pollutants = ['SO_2', 'O_3', 'NO_2', 'PM10']
data_types = ['day', 'month']

for pollutant in primary_pollutants:
    ExponencialSmothingModel(pollutant, 'day', ahead=2)
    ExponencialSmothingModel(pollutant, 'month', ahead=1)
