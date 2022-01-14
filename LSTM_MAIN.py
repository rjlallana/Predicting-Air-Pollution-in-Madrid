# LSTM_main.py
# references: https://www.tensorflow.org/tutorials/structured_data/time_series
# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
# https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

from keras.backend import clear_session
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential, load_model 
from keras.layers import Dense, LSTM, SimpleRNN, Dropout, Activation
from keras.optimizers import RMSprop

from keras.callbacks import ModelCheckpoint, EarlyStopping

# plt.rcParams["figure.figsize"] = (20,10)

def LSTM_model(target_pollutant, data_type, ahead, n_lag, TRAIN_MODEL=False):
    file_name = target_pollutant+data_type+str(ahead)
    print('Predicting pollutant ' + target_pollutant + ' for ' + str(ahead) + ' ' + data_type + ' ahead')
    data = pd.read_csv('madrid_air_quaility_2001-2018.csv', sep=',', header=0,  index_col='date', parse_dates=True)
    station_data = pd.read_csv('stations.csv', sep=',', header=0, index_col=0)
    valid_stations_list = station_data.index.to_list()
    data = data[data.station.isin(valid_stations_list)]

    if data_type == 'day':
        data = data[target_pollutant].fillna(0).resample('D').mean()
    else:
        data = data[target_pollutant].dropna().resample('D').mean().resample('M').max()

    split_year = '2016'

    train = data[data.index < split_year]
    test = data[data.index >= split_year]

    X_train = train.values.reshape(-1,1)
    X_test  = test.values.reshape(-1,1)

    if ahead > 1:
        y_train = train.shift(-(ahead-1))
        y_test  = test.shift(-(ahead-1))

        y_test = y_test[:-(ahead-1)] 
        y_train = y_train[:-(ahead-1)]

        X_test = X_test[:-(ahead-1)] 
        X_train = X_train[:-(ahead-1)]

        dates_test = y_test.index[n_lag:]
        dates_train = y_train.index[n_lag:]

        y_test = y_test.values
        y_train = y_train.values
 
    else:
        y_train = train.values
        y_test  = test.values
        dates_test = test.index[n_lag:]
        dates_train = train.index[n_lag:]


    y_train = y_train.reshape(-1,1)
    y_test  = y_test.reshape(-1,1)    

    # normalize the data to 0, 1
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = scaler.transform(y_train)
    y_test = scaler.transform(y_test)

    # The RNN takes input data with the shape  [batch_size, time_steps, Features]
    # 
    if data_type=='day':
        batch_size = 32 # input tensor shape [32, 30, 1]
    else:
        batch_size = 1 # input tensor shape [4, 12, 1]
    split_validation = int(X_train.size*0.67) 
    train_data_gen = TimeseriesGenerator(data=X_train[:split_validation], targets=y_train[:split_validation], length=n_lag, batch_size=batch_size) # batch_size=4 for montly data
    validation_data_gen = TimeseriesGenerator(data=X_train[split_validation:], targets=y_train[split_validation:], length=n_lag, batch_size=batch_size) # batch_size=4 for montly data
    test_data_gen  = TimeseriesGenerator(data=X_test, targets=y_test, length=n_lag, batch_size=1)

    def build_LSTM():
        model = Sequential()
        model.add(LSTM(32, return_sequences=True, input_shape=(n_lag, 1)))
        model.add(LSTM(16, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer=RMSprop(clipvalue=1.0), loss='mae')
        return model

    model = build_LSTM()
    num_epochs = 50

    plt.style.use('bmh')

    def plot_loss(history, title):
        plt.figure(figsize=(10,6))
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title(title)
        plt.xlabel('Nb Epochs')
        plt.ylabel('Loss')
        plt.legend()
        # plt.savefig('figures/LSTM/history'+file_name)
        # plt.close()
        plt.show()
        
        val_loss = history.history['val_loss']
        min_idx = np.argmin(val_loss)
        min_val_loss = val_loss[min_idx]

        print('Minimum validation loss of {} reached at epoch {}'.format(min_val_loss, min_idx))
    
    if TRAIN_MODEL:
        print('Start training')
        t0 = time.time()
        # step per epoch is equal to the training data size divided by the batch size
        # the data is feed into the network in temporal order
        model_history = model.fit_generator(train_data_gen
                                            , epochs=num_epochs
                                            , validation_data=validation_data_gen
                                            , verbose=1
                                            , shuffle=False
                                            # , callbacks=[EarlyStopping(monitor='loss', mode='min')]  # training will stop when the quantity monitored has stopped decreasing
        )
        t1 = time.time()
        plot_loss(model_history, 'Train & Validation Loss')
        training_time = t1-t0
        t_sec = str(training_time%60)
        t_min = str(int(training_time/60))
        # model.save('weights/'+file_name+'.hdf5')
        print('The training lasted '+ t_min +' min and '+t_sec+'seconds')

    else:
        model = load_model('weights/'+file_name+'.hdf5')
        print('Netowrk Architecture', model.summary())


    def plot_prediction(prediction, true_val, dates):
        true_val =  scaler.inverse_transform(true_val)
        true_val = true_val[n_lag:]
        prediction = scaler.inverse_transform(prediction.reshape(-1, 1))
        pred = pd.DataFrame(prediction, index=dates[0:prediction.size])
        true_val = pd.DataFrame(true_val, index=dates[0:prediction.size])

        print('Model evaluation')
        print('R2', round(r2_score(true_val, prediction), 4))
        print('MSE', round(mean_squared_error(true_val, prediction), 4))
        error = abs(prediction - true_val.values)
        print('Mean absolute error', round(error.mean(), 4))

        pollutant_color = {'SO_2':'limegreen', 'O_3':'dodgerblue', 'NO_2':'magenta', 'PM10':'blueviolet'}
        tittle = 'Predicting pollutant ' + target_pollutant + ' for ' + str(ahead) + ' ' + data_type + ' ahead'
        plt.figure(title)
        plt.plot(dates, true_val, color='lightcoral')
        plt.plot(dates, prediction, color=pollutant_color.get(target_pollutant))    
        plt.ylabel(target_pollutant+' (\u03BCg/m\u00b3)')
        plt.legend(labels=['True value', 'Prediction'])
        # plt.savefig('figures/LSTM/'+file_name)
        # plt.close()
        plt.show()
        
    print('Evalation for Test')
    pred = model.predict_generator(test_data_gen)
    plot_prediction(pred, y_test, dates_test)
    clear_session() # release the model from memory

primary_pollutants = ['SO_2', 'O_3', 'NO_2', 'PM10']

for pollutant in primary_pollutants:
    LSTM_model(pollutant, 'day', ahead=1, n_lag=30, TRAIN_MODEL=False)
    LSTM_model(pollutant, 'day', ahead=2, n_lag=30, TRAIN_MODEL=False)
    LSTM_model(pollutant, 'month', ahead=1, n_lag=12, TRAIN_MODEL=False)