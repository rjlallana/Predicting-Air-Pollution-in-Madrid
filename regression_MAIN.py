# regression_main.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# plt.rcParams["figure.figsize"] = (20,10)

def RegressionModel(target_pollutant, data_type='day', ahead=1, n_lag=15, multivariate=True, station=0):
    print('---------------------------------------------------------')
    print('------------Prediction of the pollutant: ', target_pollutant+'-----------')
    print('---------------------------------------------------------')
    split_year = '2016'
    print('Parametters:\n\
    ahead = {}\n\
    n_lag = {}\n\
    multivariate = {}'.format(ahead,n_lag,multivariate))


    data = pd.read_csv('madrid_air_quaility_2001-2018.csv', sep=',', header=0,  index_col='date', parse_dates=True)
    station_data = pd.read_csv('stations.csv', sep=',', header=0, index_col=0)

    valid_stations_list = station_data.index.to_list()
    data = data[data.station.isin(valid_stations_list)]
    def get_features(data, target_pollutant):
        D = data
        D.insert(0,'day',D.index.day)         
        D.insert(0,'month',D.index.month)         
        D.insert(0,'year',D.index.year)

        correlation_matrix = D.corr()[target_pollutant]

        strong_correlation = correlation_matrix[correlation_matrix >= 0.5]
        strong_correlation = strong_correlation.dropna(axis='rows', how='all')
        print('Correaltion higher than 0.5:')
        print(strong_correlation)

        features = strong_correlation.dropna()
        features = features.index.to_list()

        column_list = D.columns.to_list()
        features.remove(target_pollutant)

        return features
    if station > 0:
        data = data[data.station == station]
    if multivariate:
        features = get_features(data, target_pollutant)
    else:
        features = []

    if data_type=='day':
        D = pd.DataFrame(data[target_pollutant].dropna().resample('D').mean())
    else:
        D = pd.DataFrame(data[target_pollutant].dropna().resample('D').mean().resample('M').max())

    if features:
        if data_type=='day':
            temp = data[features].fillna(0).resample('D').mean()
        else:
            temp = data[features].fillna(0).resample('D').mean().resample('M').max()
        for f in features:
            D.insert(D.shape[1], f, temp[f])
    
    features = [target_pollutant] + features

    for shift in range(n_lag):
        for f in features:
            D['t-'+str(shift+1)+'_'+f] = D[f].shift((shift+1))

    # target
    D['t+'+str(ahead)+'_'+target_pollutant] = D[target_pollutant].shift(-ahead)

    D = D.fillna(0)
    if station > 0:
        split_year = str(D.index.max().year - 2)

    X = pd.DataFrame(D, index=D.index)
    X = X.drop('t+'+str(ahead)+'_'+target_pollutant, axis=1)
    X = X[n_lag:]
    y = pd.DataFrame(D['t+'+str(ahead)+'_'+target_pollutant], index=D.index)
    y = y[n_lag:]

    X_train = X[X.index <  split_year]
    X_test =  X[X.index >= split_year]

    y_train = y[y.index <  split_year]
    y_test =  y[y.index >= split_year]

    y_test = y_test[:-ahead] 
    X_test = X_test[:-ahead] 

    print(split_year)
    print('Features: ', X_train.columns.to_list())
    print('Target: ', y_train.columns.to_list())

    model = LinearRegression()
    model.fit(X_train, y_train.values)

    prediction = model.predict(X_test)
    print('Model evaluation')
    print('R2', round(r2_score(y_test, prediction), 4))
    print('MSE', round(mean_squared_error(y_test, prediction), 4))
    error = abs(prediction - y_test.values)
    print('Mean absolute error', round(error.mean(), 4))
    date_range = y_test.index

    plt.style.use('bmh')
    pollutant_color = {'SO_2':'limegreen', 'O_3':'dodgerblue', 'NO_2':'magenta', 'PM10':'blueviolet'}
    
    plt.figure(pollutant+data_type+' ahead '+str(ahead)+' prediction')
    plt.plot(date_range, y_test, color='lightcoral')
    plt.plot(date_range, prediction, color=pollutant_color.get(target_pollutant))
    plt.legend(labels=['True value', 'Prediction'])
    max_val = max(max(prediction), max(y_test.values))
    plt.yticks(np.arange(0, max_val+10, 10))
    plt.ylabel(target_pollutant+' (\u03BCg/m\u00b3)')
    # plt.savefig('figures/linear_regression/'+target_pollutant+data_type+str(ahead))
    # plt.close()
    plt.show()

primary_pollutants = ['SO_2', 'O_3', 'NO_2', 'PM10']

for pollutant in primary_pollutants:
    RegressionModel(pollutant, 'day', ahead=1, n_lag=15)
    RegressionModel(pollutant, 'day', ahead=2, n_lag=15)
    RegressionModel(pollutant, 'month', ahead=1, n_lag=12)
