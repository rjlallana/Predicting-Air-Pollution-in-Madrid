import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### DATA PREPARATION

data = pd.read_csv('madrid_air_quaility_2001-2018.csv', sep=',', header=0,  index_col='date', parse_dates=True)
station_data = pd.read_csv('stations.csv', sep=',', header=0, index_col=0)

valid_stations_list = station_data.index.to_list()
data = data[data.station.isin(valid_stations_list)]
data.sort_index(inplace=True)
# AQI
primary_pollutants = ['SO_2', 'O_3','NO_2', 'PM10']
levels = ['Good', 'Fair', 'Moderate', 'Poor', 'Very Poor']
AQI = pd.DataFrame(index=levels, columns=primary_pollutants)
AQI.SO_2 = [100,200,350,500,750]
AQI.O_3  = [50, 100, 130, 240, 380]
AQI.NO_2 = [40, 90, 120, 230, 340]
AQI.PM10 = [20, 40, 50, 100, 150]
colors = ['aquamarine', 'mediumaquamarine', 'yellow', 'orangered', 'red']
pollutant_color = {'SO_2':'limegreen', 'O_3':'dodgerblue', 'NO_2':'magenta', 'PM10':'blueviolet'}

data = data[data.index < '2018']

style = 'bmh'
plt.style.use(style)

nan_values = data[primary_pollutants].isna().sum()

plt.bar(x=primary_pollutants, height=nan_values.values, data=nan_values.values, color=pollutant_color.values())
plt.ylim(0, data.shape[0])
plt.show()

figures = []

for pollutant in primary_pollutants:
    d = data[pollutant].dropna().resample('D').max()
    figures.append(d.plot.density(color=pollutant_color.get(pollutant)))

plt.title('Density of pollutant ' + pollutant)
plt.xlabel('\u03BCg/m\u00b3')
plt.ylabel('Density')
plt.legend(labels=primary_pollutants)
# plt.savefig(fname='figures/Denisity')
plt.show()

fig, axs = plt.subplots(4)
i = 0
# daily mean
for pollutant in primary_pollutants:
    d = data[pollutant].dropna().resample('D').mean()
    max_val = d.max() + 10
    axs[i].set_ylabel(pollutant+' (\u03BCg/m\u00b3)')
    aqi_lines = AQI[AQI<=max_val][pollutant].dropna().values
    axes = []
    axs[i].scatter(d.index, d, color=pollutant_color.get(pollutant), s=8)
    for line in range(aqi_lines.size):
        axes.append(axs[i].axhline(y=AQI[pollutant][line],color=colors[line],linewidth=1.75))
    i += 1
    
if axes:
    plt.legend(loc='best', handles = axes, labels = levels)
plt.show()

fig, axs = plt.subplots(4)
i = 0
# Monthly mean of the daily maximum values.
for pollutant in primary_pollutants:
    d = data[pollutant].dropna().resample('D').max().resample('M').mean()
    max_val = d.max() + 10
    axs[i].set_ylabel(pollutant+' (\u03BCg/m\u00b3)')
    aqi_lines = AQI[AQI<=max_val][pollutant].dropna().values
    axes = []
    axs[i].plot(d, color=pollutant_color.get(pollutant))
    for line in range(aqi_lines.size):
        axes.append(axs[i].axhline(y=AQI[pollutant][line],color=colors[line],linewidth=1.75))
    i += 1
    
if axes:
    plt.legend(loc='best', handles = axes, labels = levels)
plt.show()

