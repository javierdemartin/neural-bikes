from Data_mgmt import Data_mgmt
from neural_model import Neural_Model
import os
import sys
from cluster import Cluster


cluster = Cluster()
cluster.do_cluster(sys.argv[1])


data  = Data_mgmt()
#data.iterate()

#data.supervised_learning()

data.split_sets(0.8, 0.15, 0.05)

m = Neural_Model()
m.fit_model()

data.prepare_tomorrow()
data.prepare_today()

m.tomorrow(append_to_db = False)

