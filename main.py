from Data_mgmt import Data_mgmt
from neural_model import Neural_Model
import os
import sys
from cluster import Cluster

# Parameters
# [1]: CITY
# [2]: INFLUX DB PASSWORD
# [3]: LOCAL DATA OR QUERY REMOTE DB

# cluster = Cluster(city=sys.argv[1])
# labels = cluster.do_cluster()
# # 
# data  = Data_mgmt()
# # dataset = data.read_dataset()
# # data.iterate(dataset = dataset, cluster_data = labels)
# data.supervised_learning()
# data.split_sets(0.7, 0.25, 0.5)

m = Neural_Model()
m.fit_model()
m.test_models_score()

# data.prepare_tomorrow(labels)
# data.prepare_today()
 
# m.tomorrow(append_to_db = False)