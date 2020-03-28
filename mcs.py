'''
    simple historical bootstrapping Monte-Carlo simulation of stock data

    use historical stock data as a basis for unbiased Monte-Carlo simulations

    usage:
    python3 msc.py <historical_data.csv> <forecast_days> <MC_steps>
'''

from tqdm import tqdm
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from datetime import datetime
from datetime import timedelta


print('usage: python3 msc.py <historical_data.csv> <forecast_days> <MC_steps>' +
    ' <plot_n_historical>')

### parse input
# historical data
try:
    datafile = str(sys.argv[1])
except:
    raise RuntimeError('please provide a filename with historical data')

# forecast timespan in days
try:
    n = int(sys.argv[2])
except:
    n = 250

# number of MC runs
try:
    samples = int(sys.argv[3])
except:
    samples = 1000

# historical days to plot
try:
    plot_hist = int(sys.argv[4])
except:
    plot_hist = 1

print('historical data: {}'.format(datafile))
print('forecast timespan: {} days'.format(n))
print('number of MC steps: {}'.format(samples))
print('adding last {} days of historical data to plot'.format(plot_hist))


### read and pepare input data
print('reading data ...')
try:
    d = pd.read_csv(datafile)
    d.drop(columns=['Open', 'High', 'Low', 'Adj Close', 'Volume'], inplace=True)
    d.dropna(inplace=True)
    d['Date'] = pd.to_datetime(d['Date'])
    d['change'] = d['Close'].pct_change()
    d.drop(index=0, inplace=True)
    d.reset_index(drop=True, inplace=True)
except:
    raise RuntimeError('something went wrong preparing the data; ' +
                        'please use exports from yahoo-finance')



# take latest stock price as starting value for forecast
latest = d['Close'].iloc[-1]

# what number of datasets can we draw the change from
len_hist = len(d)

# create empty dataframe to store forecast data
forecast_cols = ['Date',]
for i in range(samples):
    forecast_cols.append('sample_{}'.format(i))
forecast = pd.DataFrame(columns=forecast_cols)
# forecast datetime spacing based on historical data assuming even distribution
timerange = d['Date'].iloc[-1] - d['Date'].iloc[0]
timespacing = timerange / len(d)
forecast['Date'] = [d['Date'].iloc[-1] + i*timespacing for i in range(n+1)]


### make forecast
print('forecasting ...')
for i in tqdm(range(samples)):
    # draw random numbers
    draw = np.random.randint(0, len_hist, size=n)
    # initialize run
    run = np.zeros(n+1)
    run[0] = latest
    # perform prediction run
    for j in tqdm(range(n), leave=False):
        run[j+1] = run[j] * (1 + d['change'].iloc[draw[j]])
    # store results in forecast dataframe
    forecast['sample_{}'.format(i)] = run


### analyze forcast data
print('analyzing ...')
forecast['mean'] = forecast[['sample_{}'.format(i) 
    for i in range(samples)]].mean(axis=1)
forecast['median'] = forecast[['sample_{}'.format(i) 
    for i in range(samples)]].median(axis=1)
forecast['percentile01'] = forecast[['sample_{}'.format(i) 
    for i in range(samples)]].quantile(q=0.01, axis=1)
forecast['percentile10'] = forecast[['sample_{}'.format(i) 
    for i in range(samples)]].quantile(q=0.10, axis=1)
forecast['percentile25'] = forecast[['sample_{}'.format(i) 
    for i in range(samples)]].quantile(q=0.25, axis=1)
forecast['percentile75'] = forecast[['sample_{}'.format(i) 
    for i in range(samples)]].quantile(q=0.75, axis=1)
forecast['percentile90'] = forecast[['sample_{}'.format(i) 
    for i in range(samples)]].quantile(q=0.90, axis=1)
forecast['percentile99'] = forecast[['sample_{}'.format(i) 
    for i in range(samples)]].quantile(q=0.99, axis=1)


### plot the results
print('plotting ...')
plt.figure(figsize=(16,10))
plt.plot(d['Date'].iloc[-plot_hist:], d['Close'].iloc[-plot_hist:],
    color='blue', label='historic')

plt.plot(forecast['Date'], forecast['mean'],
    color='black', label='predict mean')
plt.plot(forecast['Date'], forecast['median'],
    color='orange', label='predict median')

plt.fill_between(forecast['Date'], 
    forecast['percentile01'], forecast['percentile99'],
    facecolor='green', alpha=0.1, label='98 % confidence')
plt.fill_between(forecast['Date'], 
    forecast['percentile10'], forecast['percentile90'],
    facecolor='green', alpha=0.3, label='80 % confidence')
plt.fill_between(forecast['Date'], 
    forecast['percentile25'], forecast['percentile75'],
    facecolor='green', alpha=0.5, label='50 % confidence')

plt.legend(loc='lower left')
plt.show()