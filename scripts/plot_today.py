import numpy as np
from Plotter import Plotter
import os
from Data_mgmt import Data_mgmt
from influxdb import InfluxDBClient
import datetime
from datetime import timedelta
import pandas as pd

client = InfluxDBClient('localhost', '8086', 'root', 'root', 'Bicis_Bilbao_Availability')
client_pred = InfluxDBClient('localhost', '8086', 'root', 'root', 'Bicis_Bilbao_Prediction')

d = Data_mgmt()

dir_path = os.path.dirname(os.path.realpath(__file__))

p = Plotter()
today = datetime.datetime.today() - timedelta(days=1)
yesterday = today - timedelta(days=1)
yesterday = yesterday.strftime('%Y-%m-%dT00:00:00Z')
today = today.strftime('%Y-%m-%dT00:00:00Z')


print(today)
print(yesterday)

for station in d.list_of_stations:

	query = 'select * from bikes where time > \'' + str(today) + '\' and station_name=\'' + str(station) + '\''

	data = pd.DataFrame(client.query(query, chunked=True).get_points())


	print(data.head())

	asdfa()

	query = 'select * from bikes where time > \'' + str(yesterday) + '\' and time < \'' + str(today)  + '\' and station_name=\'' + str(station) + '\''

	pred = pd.DataFrame(client_pred.query(query, chunked=True).get_points())

	if data.size == 0:
		continue


	data = data[['time', 'station_id', 'station_name', 'value']]

	data['time'] = pd.to_datetime(data['time'])
	data['station_id'] = pd.to_numeric(data['station_id'])
	data['value'] = pd.to_numeric(data['value'])

	date_str = data['time'].iloc[0].strftime('%Y-%m-%d')
	time_range = pd.date_range(date_str + ' 00:00:00+00:00', date_str + ' 23:50:00+00:00', freq='10T')

	data = data.set_index(keys=['time']).resample('10min').bfill()

	df = data.reindex(time_range, fill_value=np.NaN)
	df = df.reset_index()
	df.columns = ['time', 'station_id','station_name', 'value']

	df['time'] = pd.to_datetime(df['time'])

	df['value'] = df['value'].fillna(method='bfill')
	df['station_id'] = df['station_id'].fillna(method='bfill')
	df['station_name'] = df['station_name'].fillna(method='bfill')


	data = data.reset_index()


	p.two_plot(df["value"].values, pred["value"].values, "", "", "", dir_path + "/plots/tomorrow/" + str(station))

