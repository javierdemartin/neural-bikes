import numpy as np
import sys
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_context('talk')
import os
from Data_mgmt import Data_mgmt
from utils import Utils
import folium
import os
from kneed import DataGenerator, KneeLocator
from Plotter import Plotter
from Timer import Timer
from datetime import datetime

class Cluster:

	# Specify if the clustering has to be done
	weekday_analysis = True
	city = ""
	n_clusters = -1
	plotter = Plotter()
	dir_path = ""
	locations = ""
	position = ""
	
	Ks = range(1,11)

	def __init__(self, weekday_analysis = True, city = ""):
		self.weekday_analysis = weekday_analysis
		self.city = city
		self.dir_path = os.path.dirname(os.path.realpath(__file__))
		self.timer = Timer(city = self.city)
		
		self.utils = Utils(city=self.city)
		
		if weekday_analysis is True:
			self.type_of_analysis = "weekday"
		else:
			self.type_of_analysis = "weekend"
		
		self.utils.check_and_create(["/data/" + self.city +  "/cluster/cluster_data","/data/" + self.city +  "/cluster/cluster_data/" + self.type_of_analysis,"/plots/" + self.city +  "/cluster/"])

	def do_cluster(self):
		if os.path.isfile(self.dir_path + "/data/" + self.city +  "/cluster/cluster_stations.csv"):

			mtime = os.path.getmtime(self.dir_path + "/data/" + self.city +  "/cluster/cluster_stations.csv")

			last_modified_date = datetime.fromtimestamp(mtime)

			timeDiff = datetime.now() - last_modified_date
			
			return pd.read_csv(self.dir_path + "/data/" + self.city + "/cluster/cluster_stations.csv")
		
			# if timeDiff.days < 15:
# 				return pd.read_csv(self.dir_path + "/data/" + self.city + "/cluster/cluster_stations.csv")
# 			else:
# 				self.d = Data_mgmt(city=self.city)
# 
# 				print("> Reading dataset from DB")
# 				raw = self.d.read_dataset(no_date_split=True)
# 		
# 				self.timer.start()
# 				labels = self.cluster_analysis("weekday", raw)
# 				self.timer.stop("Cluster analysis done, found " + str(len(labels)) + " clusters/")
# 
# 				return labels
				
		else: 
			self.d = Data_mgmt(city=self.city)

			print("> Reading dataset from DB")
			raw = self.d.read_dataset(no_date_split=True)

			self.timer.start()
			labels = self.cluster_analysis("weekday", raw)
			self.timer.stop("Cluster analysis done, found " + str(len(labels)) + " clusters/")

			return labels


	# Type is weekday or weekend
	def cluster_analysis(self, type, raw_data):
	
		self.locations = self.utils.stations_from_web(city = self.city)
		self.position = [self.locations['lat'].iloc[0], self.locations['lon'].iloc[0]]
	
		max_bikes = raw_data.groupby('station_name')['value'].max()

		print("> There are " + str(max_bikes.shape[0]) + " stations")

		wrong_stations = max_bikes[max_bikes == 0].index.tolist()

		well_station_mask = np.logical_not(raw_data['station_name'].isin(wrong_stations))

		data = raw_data[well_station_mask]

		# Time resampling, get data every 5 minutes
		df = (data.set_index('time')
				.groupby('station_name')['value']
				.resample('10T')
				.mean()
				.bfill())

		df = df.unstack(0)
	
		# Daily profile getting rid out of sat and sun 
		weekday = df.index.weekday
		
		title = "Cluster analysis for " + sys.argv[1] 
		
		if type == "weekday":
			mask = weekday < 5
			title += " on weekdays"
			type_of_analysis = "weekday"
		else:
			mask = weekday > 4
			title += " on weekends"
			type_of_analysis = "weekend"
			

		df['hour'] = df.index.hour

		df = df.groupby('hour').mean()

		# normalization
		df_norm = df / df.max()
		
		# Some values vould be nil producing 
		# Input contains NaN, infinity or a value too large for dtype('float64')
		
		pd.set_option('display.max_columns', None) 
		
		df_norm = df_norm.dropna(axis=1)
		
		df_norm = df_norm.replace([np.inf, -np.inf], np.nan)
		df_norm = df_norm.fillna(df_norm.mean())
		
		df_norm.index.name = 'id'
		
		distortions = []
		
		for k in self.Ks:
			kmeanModel = KMeans(n_clusters=k)
			kmeanModel.fit(df_norm.T)
			distortions.append(kmeanModel.inertia_)
			
		kneedle = KneeLocator(self.Ks, distortions, curve='convex', direction='decreasing')

		self.n_clusters = round(kneedle.knee)
					
		plt.figure(figsize=(15,9))

		plt.xlabel('Hour')
		plt.xticks(np.linspace(0,24,13))
		plt.yticks(np.linspace(0,100,11))
		plt.ylabel("Available bikes (%)")

		plt.title(title)
		sns.despine()
		
		ax = plt.axes(frameon=True)

# 		ax.spines["top"].set_visible(False)
# 		ax.spines["bottom"].set_visible(False)
# 		ax.spines["right"].set_visible(False)
# 		ax.spines["left"].set_visible(False)

		ax.set_xlim(left = 0, right = 11)
		ax.xaxis.label.set_visible(False)
		
		plt.plot(self.Ks, distortions, 'bx-')
		plt.axvline(x=self.n_clusters, linewidth=4, color='r')
		plt.title('The Elbow Method showing the optimal k (' + str(self.n_clusters) + ")")
		plt.savefig(self.dir_path + "/plots/" + self.city +  "/cluster/elbow_method.png")
		plt.close()
		
		distortions_df = pd.DataFrame(distortions)

		distortions_df.to_csv(self.dir_path + "/data/" + self.city +  "/cluster/distortions.csv", index_label='id', header=['values'])

		
		kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(df_norm.T)
		label = pd.Series(kmeans.labels_)

		colors = sns.color_palette('bright', self.n_clusters)

		sns.palplot(colors)
		
		cluster_df = pd.DataFrame(kmeans.cluster_centers_)
		
		(cluster_df.T).to_csv(self.dir_path + "/data/" + self.city +  "/cluster/cluster_data/data.csv", index_label='id')	

		cluster_df.to_csv(self.dir_path + "/data/" + self.city +  "/cluster/" + str(self.city) + "_clusters_" + type_of_analysis +".csv", index=False)

		with sns.axes_style("darkgrid", {'xtick.major.size': 8.0}):
			fig, ax = plt.subplots(figsize=(10,6))

		for k, label, color in zip(kmeans.cluster_centers_, range(self.n_clusters), colors):
			plt.plot(100*k, color=color, label=label)

		plt.legend()
		plt.xlabel('Hour')
		plt.xticks(np.linspace(0,24,13))
		plt.yticks(np.linspace(0,100,11))
		plt.ylabel("Available bikes (%)")

		plt.title(title)
		sns.despine()
		plt.savefig(self.dir_path + "/plots/" + self.city +  "/cluster/" + str(sys.argv[1]) + "_pattern_" + type_of_analysis  + ".png")

		mask = np.logical_not(self.locations['nom'].isin(wrong_stations))

		self.locations = self.locations[mask]

		dflabel = pd.DataFrame({"label": kmeans.labels_}, index=df_norm.columns)
		

		self.locations = self.locations.merge(dflabel, right_index=True, left_on='nom')
		
		self.locations.drop_duplicates(inplace=True)

		mp = folium.Map(location=self.position, zoom_start=13, tiles='cartodbpositron')

		hex_colors = colors.as_hex()

		for _, row in self.locations.iterrows():

			folium.CircleMarker(
				location=[row['lat'], row['lon']],
				radius = 5,
				popup = row['nom'],
				color = hex_colors[row['label']],
				fill = True,
				fill_opacity = 0.5,
				foll_color = hex_colors[row['label']]
			).add_to(mp)


		mp.save(self.dir_path + "/plots/" + self.city +  "/cluster/" + str(sys.argv[1]) + "_map_" + type_of_analysis + ".html")
				
		dflabel = dflabel.reset_index()
		
		labels_dict = dict(zip(dflabel.station_name, dflabel.label))
		
		for label in dflabel.label.unique():
		
			if not os.path.exists(self.dir_path + "/data/" + self.city +  "/cluster/cluster_data/" + str(label)):
				os.makedirs(self.dir_path + "/data/" + self.city +  "/cluster/cluster_data/" + str(label))
			
			result = [k for k,v in labels_dict.items() if v == label]
			
			plt.close()
			plt.legend()
			plt.figure(figsize=(15,9))

			plt.xlabel('Hour')
			plt.xticks(np.linspace(0,24,13))
			plt.yticks(np.linspace(0,100,11))
			plt.ylabel("Available bikes (%)")

			plt.title(title)
			sns.despine()

			ax.spines["top"].set_visible(False)
			ax.spines["bottom"].set_visible(False)
			ax.spines["right"].set_visible(False)
			ax.spines["left"].set_visible(False)
			ax = plt.axes(frameon=False)

			ax.set_xlim(left = 0, right = 24)
			ax.xaxis.label.set_visible(False)

			plt.title(title + " for cluster name " + str(label))
			plt.savefig(self.dir_path + "/plots/" + self.city +  "/cluster/" + str(sys.argv[1]) + "_pattern_" + type_of_analysis + "_cluster_" + str(label)  + ".png")
			plt.close()			
	
		dflabel.to_csv(self.dir_path + "/data/" + self.city +  "/cluster/" + "cluster_stations.csv", index=False)	
		
		dflabel.to_csv(self.dir_path + "/data/" + self.city +  "/cluster/" + str(sys.argv[1]) + "-stations-" + type_of_analysis + ".csv", index=False)

		return dflabel