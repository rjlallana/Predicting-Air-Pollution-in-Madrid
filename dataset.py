import numpy as np
import pandas as pd

### Station data

# station_data = pd.read_csv('stations.csv', sep=',', header=0, index_col=0)
# station_data.index = station_data.index - 2807900

# # Madrid town hall divide these stations in 5 big areas 
# # http://www.mambiente.madrid.es/opencms/export/sites/default/calaire/Anexos/zonas_madrid.pdf
# # Add this data as a new column

# # 28079 000
# # 28079 code for the city of madrid
# # last 3 digits is the actual station code

# station_data = station_data.insert(0, 'area', 0)

# area = {
#     '1': [8,47,50,11,38,4,39,35,48,49],
#     '2': [36, 40,54],
#     '3': [27, 16, 55, 57, 59, 60],
#     '4': [24, 58],
#     '5': [17, 18, 56]
# }

# for i in station_data.index:
#     station_area = 0
#     for j in list(area):
#         if i in area.get(j):
#             station_area = j
#     station_data.loc[i, 'area'] = int(station_area)

# station_data.to_csv('stations.csv')

### Main dataset
station_data = pd.read_csv('stations.csv', sep=',', header=0, index_col=0)

# Csv data taken from: https://www.kaggle.com/decide-soluciones/air-quality-madrid
data_frames = []
for i in range(2001, 2019):
    year = str(i)
    file_name = 'data/madrid_' + year + '.csv'
    data_frames.append(pd.read_csv(file_name, sep=','))

data = pd.concat(data_frames)
data.station = data.station - 28079000

valid_stations_list = station_data.index.to_list()
data = data[data.station.isin(valid_stations_list)]

index = np.arange(0, data.shape[0], 1)
data.set_index(index, inplace=True)
data.date = pd.to_datetime(data.date)

primary_pollutants = ['SO_2', 'O_3','NO_2', 'PM10']
order = ['station'] + primary_pollutants

rest = []
for i in data.columns.to_list():
    if i not in order:
        rest.append(i)

order = order + rest
data = data[order]

print('Saving...')
data.to_csv('madrid_air_quaility_2001-2018.csv', index=False)

